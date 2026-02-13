#!/usr/bin/env python
"""Training script using custom pipeline with MM model zoo.

This script uses factory-based model loading (which internally uses MM libraries
for model architectures and pretrained weights) but implements a custom training
loop instead of relying on MMTrack's train_model() API.
"""

import argparse
import os
import time
import importlib

import torch
from mmcv import Config
from mmdet.apis import set_random_seed

# Factory for loading detector (uses MM internally)
from model.factory.utils import load_detector_from_checkpoint
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train probabilistic detector')
    parser.add_argument('--config', required=True, help='Model config file path (MM config for architecture)')
    parser.add_argument('--checkpoint', default=None, help='Pretrained checkpoint to resume from')
    parser.add_argument('--work-dir', required=True, help='Directory to save logs and checkpoints')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate override (defaults to config optimizer lr)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--dataloader-factory', required=True,
                        help='Dataloader factory module name (e.g., mot17_train_dataloader)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--bev-pool-backend', default=None, choices=['auto', 'cuda_ext', 'torch'],
                        help='Override RCBEVDet BEV pooling backend from config')
    return parser.parse_args()


def import_dataloader_factory(factory_name: str):
    """Dynamically import dataloader factory from data.dataloaders/."""
    module_path = f"data.dataloaders.{factory_name}"
    try:
        module = importlib.import_module(module_path.replace(".py", ""))
        return module.factory
    except ModuleNotFoundError:
        raise ValueError(f"Factory '{factory_name}' not found in data.dataloaders/")
    except AttributeError:
        raise ValueError(f"Factory '{factory_name}' does not define a 'factory' function")


def build_optimizer(detector: torch.nn.Module, cfg: Config, lr_override: float | None):
    """Build optimizer from config, with optional CLI lr override."""
    opt_cfg = cfg.get('optimizer', None)
    if opt_cfg is None:
        lr = 1e-4 if lr_override is None else lr_override
        optimizer = torch.optim.AdamW(detector.parameters(), lr=lr)
        return optimizer, f"AdamW(lr={lr}) [default]"

    opt_cfg = dict(opt_cfg)
    opt_type = str(opt_cfg.pop('type', 'AdamW'))
    cfg_lr = float(opt_cfg.pop('lr', 1e-4))
    lr = cfg_lr if lr_override is None else lr_override

    if opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(detector.parameters(), lr=lr, **opt_cfg)
    elif opt_type == 'Adam':
        optimizer = torch.optim.Adam(detector.parameters(), lr=lr, **opt_cfg)
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(detector.parameters(), lr=lr, **opt_cfg)
    else:
        raise ValueError(f"Unsupported optimizer type in config: {opt_type}")
    return optimizer, f"{opt_type}(lr={lr})"


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.work_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=True)
        print(f"[INFO] Set random seed to {args.seed}")

    # Load model config (MM config for architecture definition)
    print(f"[INFO] Loading config from {args.config}")
    cfg = Config.fromfile(args.config)
    if args.bev_pool_backend is not None:
        model_cfg = cfg.model.detector if 'detector' in cfg.model else cfg.model
        if hasattr(model_cfg, "img_view_transformer"):
            model_cfg.img_view_transformer.bev_pool_backend = args.bev_pool_backend
            print(f"[INFO] Overriding bev_pool_backend={args.bev_pool_backend}")
        else:
            print("[WARN] --bev-pool-backend provided but model has no img_view_transformer; ignoring.")

    # Build detector using MM model zoo
    if args.checkpoint:
        print(f"[INFO] Loading detector from checkpoint: {args.checkpoint}")
        detector = load_detector_from_checkpoint(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    else:
        print("[INFO] Building detector from config (no checkpoint)")
        # Extract detector config from full config
        detector_cfg = cfg.model.detector if 'detector' in cfg.model else cfg.model
        detector = build_detector(detector_cfg)
        detector.init_weights()
        detector.to(device)

    detector.train()
    print(f"[INFO] Detector initialized with {sum(p.numel() for p in detector.parameters())} parameters")

    # Setup optimizer
    optimizer, optimizer_desc = build_optimizer(detector, cfg, args.lr)
    print(f"[INFO] Optimizer: {optimizer_desc}")

    # Setup dataloader
    print(f"[INFO] Loading dataloader factory: {args.dataloader_factory}")
    dataloader_factory = import_dataloader_factory(args.dataloader_factory)
    train_dataloader = dataloader_factory()
    print(f"[INFO] Dataloader created with {len(train_dataloader)} batches")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (imgs, targets) in enumerate(train_dataloader):
            imgs = imgs.to(device)

            # Forward pass
            optimizer.zero_grad()

            # Use detector's forward_train if available (MM-style training)
            if hasattr(detector, 'forward_train'):
                # Extract ground truth from targets
                gt_bboxes = [t.get('gt_bboxes', torch.empty(0, 4).to(device)) for t in targets]
                gt_labels = [t.get('gt_labels', torch.empty(0).to(device)) for t in targets]
                img_metas = [t.get('img_metas', {}) for t in targets]

                losses = detector.forward_train(
                    imgs,
                    img_metas=img_metas,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels
                )
                loss = torch.stack(
                    [v if isinstance(v, torch.Tensor) else torch.tensor(v, device=device) for v in losses.values()]
                ).sum()
            else:
                # Custom forward for detectors without forward_train
                # Note: This branch needs to be customized based on your detector
                raise NotImplementedError(
                    "Custom training loop for detectors without forward_train is not yet implemented. "
                    "Please ensure your detector has a forward_train method."
                )

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start
        print(f"\n[EPOCH {epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.work_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': detector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': cfg.pretty_text,
            }, checkpoint_path)
            print(f"[CHECKPOINT] Saved: {checkpoint_path}\n")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete! Total time: {total_time/3600:.2f} hours")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
