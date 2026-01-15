# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import pickle
import cv2
import numpy as np

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import set_random_seed

from mmtrack.core import setup_multi_processes
from mmtrack.datasets import build_dataset
from mmtrack.utils import build_ddp, build_dp, get_device

from core import *
from model import *


def parse_args():
    parser = argparse.ArgumentParser(description='mmtrack test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument('--eval', type=str, nargs='+', help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--result-dir', type=str, default=None,
        help='directory where all of annotation .txt results are saved.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if cfg.get('USE_MMDET', False):
        from core.inference import mmdet_single_gpu_test as single_gpu_test
        from mmdet.apis import multi_gpu_test
        from mmdet.datasets import build_dataloader
        from mmdet.models import build_detector as build_model
        if 'detector' in cfg.model:
            cfg.model = cfg.model.detector
    elif cfg.get('TRAIN_REID', False):
        #TODO: make sure mmdet_single_gpu_test works for re-id
        from core.inference import mmdet_single_gpu_test as single_gpu_test
        from mmdet.apis import multi_gpu_test
        from mmdet.datasets import build_dataloader

        from mmtrack.models import build_reid as build_model
        if 'reid' in cfg.model:
            cfg.model = cfg.model.reid
    else:
        from core.inference import mmtrack_single_gpu_test as single_gpu_test
        from mmtrack.apis import multi_gpu_test
        from mmtrack.datasets import build_dataloader
        from mmtrack.models import build_model
        print("**Using MMTrack dataloader")
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set random seeds. Force setting fixed seed and deterministic=True in SOT
    # configs
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('seed', None) is not None:
        set_random_seed(
            cfg.seed, deterministic=cfg.get('deterministic', False))
    cfg.data.test.test_mode = True

    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, 
                             f"{cfg.get('eval_json', 'eval')}_{timestamp}.log.json")

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    cfg.device = get_device() if cfg.get('device',
                                         None) is None else cfg.device

    # build the model and load checkpoint
    if cfg.get('test_cfg', False):
        print("**Building model with test_cfg")
        model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        print("**No test_cfg found in cfg, using default test_cfg in model")
        model = build_model(cfg.model)
    # We need call `init_weights()` to load pretained weights in MOT task.
    model.init_weights()
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    print("args.checkpoint:", args.checkpoint)
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(
            model, args.checkpoint, map_location='cpu')
        if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']

    if not hasattr(model, 'CLASSES'):
        model.CLASSES = dataset.CLASSES

    if args.fuse_conv_bn:
        Print('**Fusing conv and bn...')
        model = fuse_conv_bn(model)

    #? Check if output already exists
    analysis_cfg = cfg.get('analysis_cfg', {})
    print("=== DEBUG: Final analysis_cfg ===")
    print(analysis_cfg)
    if analysis_cfg.get('save_dir', None) is not None:
        analysis_cfg['save_dir'] = osp.join(args.work_dir, 
                                            analysis_cfg['save_dir'])
    try:
        print("Load existing output file:", args.out)
        with open(args.out, 'rb') as f:
            outputs = pickle.load(f)
            # print("Output file")
            # print(outputs)
    except FileNotFoundError:
        outputs = None
    
    if outputs is None or args.show_dir is not None:
        if not distributed:
            model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
            print("**Single GPU test")
            outputs = single_gpu_test(
                model,
                data_loader,
                outputs,
                args.show,
                args.show_dir,
                show_score_thr=args.show_score_thr,
                analysis_cfg=analysis_cfg)
        else:
            print("**Multi GPU test")
            model = build_ddp(
                model,
                cfg.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False)

            # In multi_gpu_test, if tmpdir is None, some tesnors
            # will init on cuda by default, and no device choice supported.
            # Init a tmpdir to avoid error on npu here.
            if cfg.device == 'npu' and args.tmpdir is None:
                args.tmpdir = './npu_tmpdir'

            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                    args.gpu_collect)
    # print("**Right before writing results")
    rank, _ = get_dist_info()
    # print("Rank:", rank)
    if rank == 0:
        if args.out and not os.path.exists(args.out):
            print(f'\nwriting results to {args.out}')
            # print(f"Output keys: {list(outputs.keys())}")
            print(f"Output lengths: {[len(v) for v in outputs.values()]}")
            print()
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            # print(f"Dataset type {type(dataset)}")
            format_result_tuple = dataset.format_results(outputs, **kwargs)
            res_root, resfiles, seq_names, tmp_dir = format_result_tuple

            from pathlib import Path
            # Set this to the directory where your tracking results are saved
            track_dir = Path(resfiles['track'])  # adjust if needed

            # make a copy of the annotation .txt from track_dir to specified results dir 
            print("args.result_dir:", args.result_dir)
            if args.result_dir is not None:
                print("**Copying all result .txt files to", args.result_dir)
                os.makedirs(args.result_dir, exist_ok=True)
                for track_file in track_dir.glob("*.txt"):
                    dest_file = Path(args.result_dir) / track_file.name
                    print(f"Copying {track_file} to {dest_file}")
                    with open(track_file, 'r') as src, open(dest_file, 'w') as dst:
                        dst.write(src.read())

            # -- added generate resfiles for video annotations --
            annotate_results_from_txt(
                dataset.img_prefix,        # path to MOT17 images root
                resfiles['track'],         # folder of .txt result files
                seq_names,                 # list of video names like "MOT17-10-SDP"
                output_dir='./annotated_videos'  # destination
            )
            
            # -- debug print to check frame ranges in each result file --
            # total_frames = 0
            # for track_file in track_dir.glob("*.txt"):
            #     frames = []
            #     with open(track_file) as f:
            #         last_frame = -1
            #         num_detections_for_frame = 0
            #         for line in f:
            #             if line.strip():
            #                 frame_id = int(line.strip().split(',')[0])
            #                 frames.append(frame_id)
            #                 if frame_id == last_frame:
            #                     num_detections_for_frame += 1
            #                 else:
            #                     if last_frame != -1:
            #                         print(f"  Frame {last_frame} had {num_detections_for_frame} detections")
            #                     num_detections_for_frame = 1
            #                     last_frame = frame_id
            #     if frames:
            #         # print(f"{track_file.name}: min_frame = {min(frames)}, max_frame = {max(frames)}, total unique frames = {len(set(frames))}")
            #         total_frames += len(set(frames))
            #     else:
            #         print(f"{track_file.name}: No frames")
            # print(f"Total frames across all sequences: {total_frames}")

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            if eval_kwargs:
                # hard-code way to remove EvalHook args
                eval_hook_args = [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'by_epoch', 'out_dir'
                ]
                for key in eval_hook_args:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                metric = dataset.evaluate(outputs, **eval_kwargs)
                print(metric)
                metric_dict = dict(
                    config=args.config, mode='test', epoch=cfg.total_epochs)
                metric_dict.update(metric)


# -- added function to annotate video from tracking results --
from tqdm import tqdm
def annotate_results_from_txt(dataset_root, resfile_dir, seq_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for seq_name in seq_names:
        gt_file = os.path.join(dataset_root, seq_name, 'gt', 'gt.txt')
        img_dir = os.path.join(dataset_root, seq_name, 'img1')
        track_file = os.path.join(resfile_dir, f'{seq_name}.txt')
        # print(f"[>] Annotating {seq_name}, GT: {gt_file}, Track: {track_file}, img_dir: {img_dir}")

        # Load GT
        gt_dict = {}
        with open(gt_file) as f:
            for line in f:
                frame_id, _, x, y, w, h, _, _, _ = map(float, line.strip().split(','))
                gt_dict.setdefault(int(frame_id), []).append([x, y, w, h])

        # Load tracking results
        track_dict = {}
        with open(track_file) as f:
            for line in f:
                frame_id, track_id, x, y, w, h, score, *_ = map(float, line.strip().split(','))
                track_dict.setdefault(int(frame_id), []).append([x, y, w, h, track_id])

        # Get image size and number of frames
        image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        if not image_files:
            continue
        num_frames = len(image_files)
        H, W = cv2.imread(os.path.join(img_dir, image_files[0])).shape[:2]

        # Setup writer
        out_path = os.path.join(output_dir, f"{seq_name}_annotated.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

        # === Annotate ===
        for frame_id in tqdm(range(1, num_frames + 1), desc=f"[{seq_name}]"):

            img_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)

            # Draw GT (green)
            for x, y, w, h in gt_dict.get(frame_id, []):
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, 'GT', (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw Tracks (red)
            for x, y, w, h, track_id in track_dict.get(frame_id, []):
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f'ID {int(track_id)}', (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            writer.write(img)

        writer.release()
        print(f"[✓] Saved: {out_path}")

if __name__ == '__main__':
    main()
