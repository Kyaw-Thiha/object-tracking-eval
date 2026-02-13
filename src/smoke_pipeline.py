#!/usr/bin/env python
"""Generic smoke runner for eval and train integration checks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, cast

import torch
from mmcv import Config
from mmdet.models import build_detector
from model.factory.utils import load_detector_from_checkpoint

# Keep behavior consistent when invoked from repo root or src/.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


ALLOWED_TRACKERS = ["none", "uncertainty_tracker", "probabilistic_byte_tracker", "prob_ocsort_tracker", "rcbevdet_3d_tracker"]
ALLOWED_2D_TRACKERS = ["uncertainty_tracker", "probabilistic_byte_tracker", "prob_ocsort_tracker"]


@dataclass
class SmokeReport:
    mode: str
    ok: bool
    num_batches: int
    errors: list[str]
    warnings: list[str]
    stats: dict[str, Any]


def import_dataloader(factory_name: str):
    module_path = f"data.dataloaders.{factory_name}"
    module = importlib.import_module(module_path.replace(".py", ""))
    if not hasattr(module, "factory"):
        raise ValueError(f"dataloader factory '{factory_name}' does not define factory()")
    return module.factory()


def import_model_factory(factory_name: str):
    module_path = f"model.factory.{factory_name}"
    module = importlib.import_module(module_path.replace(".py", ""))
    if not hasattr(module, "factory"):
        raise ValueError(f"model factory '{factory_name}' does not define factory()")
    return module.factory


def load_tracker(tracker_name: str):
    # Reuse tracker construction from evaluation pipeline.
    from evaluation_pipeline import load_tracker as _load_tracker

    return _load_tracker(tracker_name)


def build_tracker_model(detector_model):
    """
    Build the exact tracker-facing wrapper used by evaluation_pipeline.py.
    """
    from evaluation_pipeline import MOTDetector
    from model.kalman_filter_uncertainty import KalmanFilterWithUncertainty

    return MOTDetector(detector=detector_model, motion=KalmanFilterWithUncertainty())


def _to_bbox_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _to_label_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.long)
    return torch.as_tensor(x, dtype=torch.long, device=device)


class _SmokeBoxes3D:
    """Minimal 3D box wrapper for RCBEVDet heads expecting MM3D-style boxes."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.gravity_center = tensor[:, :3]


def _to_boxes3d_object(x: Any, device: torch.device) -> Any:
    if hasattr(x, "gravity_center") and hasattr(x, "tensor"):
        return x
    tensor = _to_bbox_tensor(x, device)
    if tensor.ndim != 2 or tensor.shape[1] < 7:
        raise ValueError(f"gt_bboxes_3d must be NxD with D>=7, got {tuple(tensor.shape)}")
    return _SmokeBoxes3D(tensor)


def _reduce_loss(v: Any, device: torch.device) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.mean()
    if isinstance(v, (float, int)):
        return torch.tensor(float(v), device=device)
    if isinstance(v, (list, tuple)):
        parts = [_reduce_loss(x, device) for x in v]
        return torch.stack(parts).sum() if parts else torch.tensor(0.0, device=device)
    raise TypeError(f"unsupported loss value type: {type(v)}")


def _build_optimizer_from_cfg(detector: torch.nn.Module, cfg: Config, lr_override: float | None):
    """Build optimizer from config, with optional CLI lr override."""
    opt_cfg = cfg.get("optimizer", None)
    if opt_cfg is None:
        lr = 1e-4 if lr_override is None else lr_override
        return torch.optim.AdamW(detector.parameters(), lr=lr)

    opt_cfg = dict(opt_cfg)
    ignored_keys: list[str] = []
    for mm_only_key in ("paramwise_cfg", "_delete_"):
        if mm_only_key in opt_cfg:
            ignored_keys.append(mm_only_key)
            opt_cfg.pop(mm_only_key)

    if ignored_keys:
        print(
            "[SMOKE] Warning: ignoring MM-only optimizer keys in smoke mode: "
            + ", ".join(ignored_keys)
        )

    opt_type = str(opt_cfg.pop("type", "AdamW"))
    cfg_lr = float(opt_cfg.pop("lr", 1e-4))
    lr = cfg_lr if lr_override is None else lr_override

    if opt_type == "AdamW":
        return torch.optim.AdamW(detector.parameters(), lr=lr, **opt_cfg)
    if opt_type == "Adam":
        return torch.optim.Adam(detector.parameters(), lr=lr, **opt_cfg)
    if opt_type == "SGD":
        return torch.optim.SGD(detector.parameters(), lr=lr, **opt_cfg)
    raise ValueError(f"Unsupported optimizer type in config: {opt_type}")


def _check_eval_target_contract(target: dict[str, Any], strict: bool, errors: list[str]):
    if not strict:
        return
    for key in ("frame_id", "video_id", "img_metas"):
        if key not in target:
            errors.append(f"target missing required key '{key}'")
    if "img_metas" in target and isinstance(target["img_metas"], dict):
        if "scale_factor" not in target["img_metas"]:
            errors.append("target['img_metas'] missing 'scale_factor'")


def _validate_model_outputs(
    batch_bboxes: Any,
    batch_labels: Any,
    batch_covs: Any,
    batch_size: int,
    strict: bool,
    errors: list[str],
) -> tuple[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if not all(isinstance(x, list) for x in (batch_bboxes, batch_labels, batch_covs)):
        errors.append("model output components must all be lists")
        return 0, []
    if not (len(batch_bboxes) == len(batch_labels) == len(batch_covs) == batch_size):
        errors.append("model output list lengths must equal batch size")
        return 0, []

    det_count = 0
    per_frame: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for j in range(batch_size):
        b = batch_bboxes[j]
        l = batch_labels[j]
        c = batch_covs[j]
        if not (isinstance(b, torch.Tensor) and isinstance(l, torch.Tensor) and isinstance(c, torch.Tensor)):
            errors.append(f"frame {j}: expected tensors for bboxes/labels/covs")
            return det_count, per_frame
        if b.ndim != 2 or b.shape[1] < 4:
            errors.append(f"frame {j}: bboxes must be Nx5 or Nx4, got {tuple(b.shape)}")
            return det_count, per_frame
        n = b.shape[0]
        if l.ndim != 1 or l.shape[0] != n:
            errors.append(f"frame {j}: labels shape mismatch, expected ({n},), got {tuple(l.shape)}")
            return det_count, per_frame
        if c.ndim != 3 or c.shape != (n, 4, 4):
            errors.append(f"frame {j}: covs shape mismatch, expected ({n},4,4), got {tuple(c.shape)}")
            return det_count, per_frame
        if strict:
            if not torch.isfinite(b).all():
                errors.append(f"frame {j}: non-finite bbox values")
                return det_count, per_frame
            if not torch.isfinite(c).all():
                errors.append(f"frame {j}: non-finite covariance values")
                return det_count, per_frame
        det_count += n
        per_frame.append((b, l, c))
    return det_count, per_frame


def _validate_tracker_output(
    output: Any,
    frame_idx: int,
    strict: bool,
    errors: list[str],
) -> int:
    if output is None:
        return 0
    if not (isinstance(output, tuple) and len(output) == 4):
        errors.append(f"tracker frame {frame_idx}: expected tuple of len=4")
        return 0
    track_bboxes, track_covs, track_labels, track_ids = output
    if not all(isinstance(x, torch.Tensor) for x in (track_bboxes, track_covs, track_labels, track_ids)):
        errors.append(f"tracker frame {frame_idx}: outputs must be tensors")
        return 0
    if track_bboxes.ndim != 2 or track_bboxes.shape[1] < 4:
        errors.append(f"tracker frame {frame_idx}: track_bboxes must be Nx5/Nx4, got {tuple(track_bboxes.shape)}")
        return 0
    n = track_bboxes.shape[0]
    if track_covs.ndim != 3 or track_covs.shape != (n, 4, 4):
        errors.append(f"tracker frame {frame_idx}: track_covs shape mismatch, got {tuple(track_covs.shape)}")
        return 0
    if track_labels.ndim != 1 or track_labels.shape[0] != n:
        errors.append(f"tracker frame {frame_idx}: track_labels shape mismatch, got {tuple(track_labels.shape)}")
        return 0
    if track_ids.ndim != 1 or track_ids.shape[0] != n:
        errors.append(f"tracker frame {frame_idx}: track_ids shape mismatch, got {tuple(track_ids.shape)}")
        return 0
    if strict:
        if not torch.isfinite(track_bboxes).all():
            errors.append(f"tracker frame {frame_idx}: non-finite track_bboxes")
            return 0
        if not torch.isfinite(track_covs).all():
            errors.append(f"tracker frame {frame_idx}: non-finite track_covs")
            return 0
    return n


def _validate_model_outputs_3d(
    outputs_3d: Any,
    batch_size: int,
    strict: bool,
    errors: list[str],
) -> tuple[int, list[dict[str, torch.Tensor]]]:
    if not isinstance(outputs_3d, list):
        errors.append("3D model output must be list[dict]")
        return 0, []
    if len(outputs_3d) != batch_size:
        errors.append("3D model output list length must equal batch size")
        return 0, []

    total = 0
    per_frame: list[dict[str, torch.Tensor]] = []
    for i, out in enumerate(outputs_3d):
        if not isinstance(out, dict):
            errors.append(f"frame {i}: 3D output must be dict")
            return total, per_frame
        for key in ("boxes_3d", "scores_3d", "labels_3d"):
            if key not in out:
                errors.append(f"frame {i}: missing key '{key}' in 3D output")
                return total, per_frame
            if not isinstance(out[key], torch.Tensor):
                errors.append(f"frame {i}: '{key}' must be tensor")
                return total, per_frame

        boxes = out["boxes_3d"]
        scores = out["scores_3d"]
        labels = out["labels_3d"]
        if boxes.ndim != 2 or boxes.shape[1] < 7:
            errors.append(f"frame {i}: boxes_3d must be NxD with D>=7, got {tuple(boxes.shape)}")
            return total, per_frame
        n = boxes.shape[0]
        if scores.ndim != 1 or scores.shape[0] != n:
            errors.append(f"frame {i}: scores_3d shape mismatch")
            return total, per_frame
        if labels.ndim != 1 or labels.shape[0] != n:
            errors.append(f"frame {i}: labels_3d shape mismatch")
            return total, per_frame

        if "velocities" in out and out["velocities"] is not None:
            v = out["velocities"]
            if not isinstance(v, torch.Tensor) or v.ndim != 2 or v.shape[0] != n or v.shape[1] != 2:
                errors.append(f"frame {i}: velocities must be Nx2 when present")
                return total, per_frame

        if strict and (not torch.isfinite(boxes).all() or not torch.isfinite(scores).all()):
            errors.append(f"frame {i}: non-finite values in 3D outputs")
            return total, per_frame

        total += n
        per_frame.append(out)

    return total, per_frame


def _validate_tracker_output_3d(
    output: Any,
    frame_idx: int,
    strict: bool,
    errors: list[str],
) -> int:
    if not isinstance(output, dict):
        errors.append(f"3D tracker frame {frame_idx}: expected dict output")
        return 0
    for key in ("boxes_3d", "scores_3d", "labels_3d", "track_ids"):
        if key not in output or not isinstance(output[key], torch.Tensor):
            errors.append(f"3D tracker frame {frame_idx}: missing/invalid '{key}'")
            return 0
    boxes = output["boxes_3d"]
    scores = output["scores_3d"]
    labels = output["labels_3d"]
    track_ids = output["track_ids"]
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        errors.append(f"3D tracker frame {frame_idx}: boxes_3d shape invalid")
        return 0
    n = boxes.shape[0]
    if scores.shape != (n,) or labels.shape != (n,) or track_ids.shape != (n,):
        errors.append(f"3D tracker frame {frame_idx}: score/label/id shape mismatch")
        return 0
    if strict and (not torch.isfinite(boxes).all() or not torch.isfinite(scores).all()):
        errors.append(f"3D tracker frame {frame_idx}: non-finite values")
        return 0
    return n


def run_eval(args) -> SmokeReport:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    if args.output_mode == "2d" and args.tracker not in ALLOWED_2D_TRACKERS and args.tracker != "none":
        errors.append(f"2D mode tracker must be one of {ALLOWED_2D_TRACKERS} or none")
        return SmokeReport(mode="eval", ok=False, num_batches=0, errors=errors, warnings=warnings, stats=stats)
    if args.output_mode == "3d" and args.tracker in ALLOWED_2D_TRACKERS:
        errors.append("3D mode supports tracker=none or tracker=rcbevdet_3d_tracker only")
        return SmokeReport(mode="eval", ok=False, num_batches=0, errors=errors, warnings=warnings, stats=stats)

    device = torch.device(args.device)
    dataloader = import_dataloader(args.dataloader_factory)
    factory = import_model_factory(args.model_factory)
    model = factory(device=args.device)
    model.eval()

    tracker = load_tracker(args.tracker) if args.tracker and args.tracker != "none" else None
    tracker_model = build_tracker_model(model) if (tracker is not None and args.output_mode == "2d") else None
    if tracker_model is not None:
        tracker_model.eval()
    if tracker is not None and hasattr(tracker, "reset"):
        tracker.reset()

    seen_batches = 0
    seen_frames = 0
    seen_dets = 0
    tracker_frames_checked = 0
    tracker_tracks_total = 0

    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break
            seen_batches += 1

            if not isinstance(imgs, torch.Tensor):
                errors.append("imgs must be torch.Tensor")
                break
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            seen_frames += batch_size

            if not isinstance(targets, list) or len(targets) != batch_size:
                errors.append("targets must be list with length == batch_size")
                break

            for t in targets:
                if not isinstance(t, dict):
                    errors.append("each target must be dict")
                    break
                _check_eval_target_contract(t, args.strict, errors)
            if errors:
                break

            if args.output_mode == "2d":
                if tracker_model is not None:
                    outputs = tracker_model(imgs, targets)
                else:
                    use_ctx = args.with_context == "on" or (args.with_context == "auto" and hasattr(model, "infer_with_context"))
                    if use_ctx:
                        if not hasattr(model, "infer_with_context"):
                            errors.append("with_context=on but model has no infer_with_context")
                            break
                        outputs = model.infer_with_context(imgs, targets)
                    else:
                        outputs = model.infer(imgs) if hasattr(model, "infer") else model(imgs)

                if not (isinstance(outputs, tuple) and len(outputs) == 3):
                    errors.append("model output must be tuple(len=3): (bboxes, labels, covs)")
                    break

                batch_bboxes, batch_labels, batch_covs = outputs
                det_count, per_frame = _validate_model_outputs(
                    batch_bboxes,
                    batch_labels,
                    batch_covs,
                    batch_size=batch_size,
                    strict=args.strict,
                    errors=errors,
                )
                seen_dets += det_count
                if errors:
                    break

                if tracker is not None:
                    for j, (det_bboxes, det_labels, det_covs) in enumerate(per_frame):
                        target = targets[j]
                        if "img_metas" not in target:
                            warnings.append(f"frame {j}: skipping tracker test (target has no img_metas)")
                            continue
                        frame_id = int(target.get("frame_id", j))
                        frame_id_tracker = frame_id - 1
                        track_output = tracker.track(
                            imgs[j],
                            img_metas=[target["img_metas"]],
                            model=tracker_model,
                            bboxes=det_bboxes,
                            bbox_covs=det_covs,
                            labels=det_labels,
                            frame_id=frame_id_tracker,
                            rescale=False,
                        )
                        tracker_frames_checked += 1
                        tracker_tracks_total += _validate_tracker_output(
                            track_output,
                            frame_idx=j,
                            strict=args.strict,
                            errors=errors,
                        )
                        if errors:
                            break
                if errors:
                    break
            else:
                if not hasattr(model, "infer_with_context_3d"):
                    errors.append("3D eval mode requires model.infer_with_context_3d")
                    break
                outputs_3d = model.infer_with_context_3d(imgs, targets)
                det_count, per_frame_3d = _validate_model_outputs_3d(
                    outputs_3d=outputs_3d,
                    batch_size=batch_size,
                    strict=args.strict,
                    errors=errors,
                )
                seen_dets += det_count
                if errors:
                    break

                if tracker is not None:
                    for j, det3d in enumerate(per_frame_3d):
                        target = targets[j]
                        frame_id = int(target.get("frame_id", j))
                        track_output = tracker.track(det3d, frame_id=frame_id)
                        tracker_frames_checked += 1
                        tracker_tracks_total += _validate_tracker_output_3d(
                            track_output,
                            frame_idx=j,
                            strict=args.strict,
                            errors=errors,
                        )
                        if errors:
                            break
                if errors:
                    break

    stats["frames_checked"] = seen_frames
    stats["detections_checked"] = seen_dets
    stats["tracker_frames_checked"] = tracker_frames_checked
    stats["tracker_tracks_total"] = tracker_tracks_total
    return SmokeReport(mode="eval", ok=len(errors) == 0, num_batches=seen_batches, errors=errors, warnings=warnings, stats=stats)


def run_train(args) -> SmokeReport:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    device = torch.device(args.device)
    cfg = Config.fromfile(args.config)

    if args.checkpoint:
        detector = load_detector_from_checkpoint(args.config, args.checkpoint, args.device)
    else:
        detector_cfg = cfg.model.detector if "detector" in cfg.model else cfg.model
        detector = build_detector(detector_cfg)
        detector.init_weights()
        detector.to(device)

    detector.train()
    optimizer = _build_optimizer_from_cfg(detector, cfg, args.lr)
    dataloader = import_dataloader(args.dataloader_factory)

    seen_batches = 0
    losses: list[float] = []

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break
        seen_batches += 1

        if not isinstance(imgs, torch.Tensor):
            errors.append("imgs must be torch.Tensor")
            break
        imgs = imgs.to(device)

        if not isinstance(targets, list):
            errors.append("targets must be list")
            break

        gt_bboxes = []
        gt_labels = []
        img_metas = []
        is_rcbevdet_batch = False
        rcbevdet_targets: list[dict[str, Any]] = []
        for t in targets:
            if not isinstance(t, dict):
                errors.append("each target must be dict")
                break
            if "img_inputs" in t and "radar_points" in t:
                is_rcbevdet_batch = True
                rcbevdet_targets.append(t)
            b = t.get("gt_bboxes", t.get("boxes", torch.zeros((0, 4), dtype=torch.float32)))
            l = t.get("gt_labels", t.get("labels", torch.zeros((0,), dtype=torch.long)))
            gt_bboxes.append(_to_bbox_tensor(b, device))
            gt_labels.append(_to_label_tensor(l, device))
            img_metas.append(t.get("img_metas", {}))
            if args.strict and "img_metas" not in t:
                errors.append("strict mode: target missing 'img_metas'")
                break
        if errors:
            break

        if not hasattr(detector, "forward_train"):
            errors.append("detector has no forward_train")
            break

        optimizer.zero_grad()
        if is_rcbevdet_batch and len(rcbevdet_targets) == len(targets):
            required_img_input_keys = ("imgs", "sensor2egos", "ego2globals", "intrins", "post_rots", "post_trans", "bda")
            missing_keys: list[str] = []
            for t in rcbevdet_targets:
                img_inputs = t.get("img_inputs", {})
                for key in required_img_input_keys:
                    if key not in img_inputs:
                        missing_keys.append(key)
            if missing_keys:
                errors.append("rcbevdet target missing img_inputs keys: " + ", ".join(sorted(set(missing_keys))))
                break

            img_inputs_batch = [
                torch.stack([cast(torch.Tensor, t["img_inputs"][k]) for t in rcbevdet_targets], dim=0).to(device)
                for k in required_img_input_keys
            ]
            radar_batch = [cast(torch.Tensor, t["radar_points"]).to(device) for t in rcbevdet_targets]
            gt_bboxes_3d = [
                _to_boxes3d_object(t.get("gt_bboxes_3d", torch.zeros((0, 9), dtype=torch.float32)), device)
                for t in rcbevdet_targets
            ]
            gt_labels_3d = [
                _to_label_tensor(t.get("gt_labels_3d", torch.zeros((0,), dtype=torch.long)), device)
                for t in rcbevdet_targets
            ]

            if any(not isinstance(m, dict) for m in img_metas):
                errors.append("rcbevdet expects img_metas as list[dict]")
                break

            # BEVDepth variants expect a depth label tensor. Use zeros when
            # dataloader does not provide dense depth supervision.
            num_frame = int(getattr(detector, "num_frame", 1))
            num_views = int(img_inputs_batch[0].shape[1])
            cams_per_frame = max(1, num_views // max(1, num_frame))
            _, _, _, h_in, w_in = img_inputs_batch[0].shape
            gt_depth = torch.zeros((len(rcbevdet_targets), cams_per_frame, h_in, w_in), dtype=torch.float32, device=device)

            loss_dict = detector.forward_train(
                points=None,
                img_metas=img_metas,
                radar=radar_batch,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                img_inputs=img_inputs_batch,
                gt_depth=gt_depth,
            )
        else:
            loss_dict = detector.forward_train(
                imgs,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
            )
        if not isinstance(loss_dict, dict) or len(loss_dict) == 0:
            errors.append("forward_train did not return a non-empty loss dict")
            break

        total_loss = torch.stack([_reduce_loss(v, device) for v in loss_dict.values()]).sum()
        if not torch.isfinite(total_loss):
            errors.append(f"non-finite loss: {float(total_loss.detach().cpu())}")
            break

        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach().cpu()))

    if losses:
        stats["loss_first"] = losses[0]
        stats["loss_last"] = losses[-1]
        stats["loss_min"] = min(losses)
        stats["loss_max"] = max(losses)
    return SmokeReport(mode="train", ok=len(errors) == 0, num_batches=seen_batches, errors=errors, warnings=warnings, stats=stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Generic smoke checks for eval/train pipelines")
    sub = parser.add_subparsers(dest="mode", required=True)

    pe = sub.add_parser("eval")
    pe.add_argument("--dataloader_factory", required=True, type=str)
    pe.add_argument("--model_factory", required=True, type=str)
    pe.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    pe.add_argument("--num_batches", default=1, type=int)
    pe.add_argument("--strict", action="store_true")
    pe.add_argument("--with_context", default="auto", choices=["auto", "on", "off"])
    pe.add_argument("--tracker", default="none", choices=ALLOWED_TRACKERS)
    pe.add_argument("--output_mode", default="2d", choices=["2d", "3d"])
    pe.add_argument("--report_json", default=None, type=str)

    pt = sub.add_parser("train")
    pt.add_argument("--dataloader_factory", required=True, type=str)
    pt.add_argument("--config", required=True, type=str)
    pt.add_argument("--checkpoint", default=None, type=str)
    pt.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    pt.add_argument("--num_batches", default=1, type=int)
    pt.add_argument("--lr", default=None, type=float)
    pt.add_argument("--strict", action="store_true")
    pt.add_argument("--report_json", default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_eval(args) if args.mode == "eval" else run_train(args)

    print(f"[SMOKE] mode={report.mode} ok={report.ok} batches={report.num_batches}")
    for w in report.warnings:
        print(f"[WARN] {w}")
    for e in report.errors:
        print(f"[ERR] {e}")
    if report.stats:
        print(f"[STATS] {report.stats}")

    if getattr(args, "report_json", None):
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2))
        print(f"[SMOKE] wrote report: {report_path}")

    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    # Keep pyright happy if imported in tooling.
    os.environ.setdefault("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED", "0"))
    main()
