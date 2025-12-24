import os
import time
import argparse

import torch
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")

import _init_path  # noqa


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='config/prob/nuscenes_gs25600_pretrain.py')
    parser.add_argument('--work-dir', type=str, default='./work_dirs/tmp')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--load-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--port', type=int, default=29500)
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    args.port = check_port(args.port)
    # print(args)
    print("=" * 66)
    print(f"config: {args.config}")
    print(f"work_dir: {args.work_dir}")
    print(f"seed: {args.seed}")
    print(f"iter_resume: {args.iter_resume}")
    print(f"gradient_accumulation: {args.gradient_accumulation}")
    print(f"gpus: {args.gpus}")

    return args

def pass_print(*args, **kwargs):
    pass

def check_port(port):
    import socket
    port = int(port)
    max_attempts = 100
    for attempt in range(max_attempts):
        current_port = port + attempt
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', current_port))
                return current_port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port in range {port}-{port + max_attempts - 1}")

def load_config(args):
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.iter_resume = args.iter_resume
    cfg.grad_accumulation = args.gradient_accumulation
    cfg.load_from = '' if not hasattr(cfg, 'load_from') else cfg.load_from
    cfg.resume_from = '' if not hasattr(cfg, 'resume_from') else cfg.resume_from
    if cfg.load_from == '' and os.path.exists(args.load_from):
        cfg.load_from = args.load_from
    if cfg.resume_from == '' and os.path.exists(args.resume_from):
        cfg.resume_from = args.resume_from
    return cfg

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = load_config(args)
    if local_rank == 0:
        print(f'batch_size: {cfg.train_loader["batch_size"]}')
        print(f'num_workers: {cfg.train_loader["num_workers"]}')
        print(f'max_epochs: {cfg.max_epochs}')
        print(f'load_from: {cfg.load_from}')
        print(f'resume_from: {cfg.resume_from}')

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", f"{args.port}")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        if local_rank == 0:
            print(f"endpoint: tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    if local_rank == 0:
        print("=" * 66)
        if os.path.exists(args.load_from) and cfg.load_from != args.load_from:
            print(f"WARNING: load_from in config is different from the one in args."
                f"Use {cfg.load_from} instead of {args.load_from}")
        if os.path.exists(args.resume_from) and cfg.resume_from != args.resume_from:
            print(f"WARNING: resume_from in config is different from the one in args."
                f"Use {cfg.resume_from} instead of {args.resume_from}")
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(os.path.join(args.work_dir, os.path.basename(args.config)))
        if os.path.islink(os.path.join(args.work_dir, 'config.py')):
            os.remove(os.path.join(args.work_dir, 'config.py'))
        os.symlink(os.path.basename(args.config),
                   os.path.join(args.work_dir, 'config.py'))
        from shutil import copyfile
        lifter_version = cfg.model.lifter.type.split('GaussianLifter')[-1].lower()
        copy_list = [
            'model/segmentor/bev_segmentor.py',
            f'model/lifter/gaussian_lifter_{lifter_version}.py',
            'model/encoder/gaussian_encoder/gaussian_encoder.py',
            'model/encoder/gaussian_encoder/deformable_module.py',
            'model/encoder/gaussian_encoder/spconv3d_module.py',
            'model/encoder/gaussian_encoder/refine_module_v3.py',
            'loss/render_loss.py',
            'loss/reproj_loss.py',
        ]
        for f in copy_list:
            if not os.path.exists(os.path.join(args.work_dir, f)):
                os.makedirs(os.path.dirname(os.path.join(args.work_dir, f)), exist_ok=True)
            copyfile(f, os.path.join(args.work_dir, f))
        from misc.tb_wrapper import WrappedTBWriter
        writer = WrappedTBWriter('gsformer', log_dir=os.path.join(args.work_dir, 'tf'))
        WrappedTBWriter._instance_dict['gsformer'] = writer
    else:
        writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('gsformer', log_file=log_file, log_level='INFO')
    MMLogger._instance_dict['gsformer'] = logger
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from tools.utils.train_utils import train_model
    gs_model = build_segmentor(cfg.model)
    if 'DistillLoss' in [loss['type'] for loss in cfg.loss.loss_cfgs]:
        gs_model.load_dinov3(model_name='vits16')
        # gs_model.load_dav2(model_name='vitb')
    gs_model.init_weights()
    n_parameters = sum(p.numel() for p in gs_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    train_model(
        gs_model,
        cfg,
        distributed,
        logger,
        local_rank=local_rank,
    )
    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    args = parse_args()
    if args.gpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
