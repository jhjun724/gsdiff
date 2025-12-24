import os
import time

import torch
import numpy as np

from mmengine.utils import symlink
from mmengine.optim import build_optim_wrapper
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler

from dataset import get_dataloader
from loss import OPENOCC_LOSS
from misc.metric_util import MeanIoU


def train_one_epoch(
    model,
    raw_model,
    loss_func,
    train_loader,
    val_loader,
    cfg,
    optimizer,
    scheduler,
    logger,
    epoch,
    global_iter,
    first_run,
    start_time,
    amp=False,
    scaler=None,
    local_rank=0,
    last_iter=0,
    metric=None,
    saved_iters=[],
    saved_epochs=[],
):
    model.train()
    os.environ['eval'] = 'false'
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)

    total_iter_per_epoch = len(train_loader)

    loss_list = []
    image_ts, lifter_ts, encoder_ts, render_ts, mvs_ts, dino_ts, head_ts = [], [], [], [], [], [], []
    time.sleep(1)
    data_time_s = time.time()
    time_s = time.time()
    elapsed_epoch = 0.0

    for i_iter, data in enumerate(train_loader):
        if first_run:
            i_iter = i_iter + last_iter

        for k in list(data.keys()):
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda()
        input_imgs = data.pop('img')
        if 'points' in data:
            input_points = data.pop('points')
        else:
            input_points = None
        data_time_e = time.time()

        with torch.cuda.amp.autocast(amp):
            # forward + backward + optimize
            result_dict = model(
                imgs=input_imgs,
                points=input_points,
                metas=data,
                global_iter=global_iter
            )

            loss_input = {
                'metas': data,
                'global_iter': global_iter
            }
            for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                loss_input.update({loss_input_key: result_dict[loss_input_val]})
            loss, loss_dict = loss_func(loss_input)
            loss = loss / cfg.grad_accumulation
        if not amp:
            loss.backward()
            if (global_iter + 1) % cfg.grad_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_max_norm
                )
                optimizer.step()
                optimizer.zero_grad()
        else:
            scaler.scale(loss).backward()
            if (global_iter + 1) % cfg.grad_accumulation == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_max_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        loss_list.append(loss.detach().cpu().item())
        scheduler.step_update(global_iter)
        time_e = time.time()

        image_ts.append(result_dict['image_time'])
        lifter_ts.append(result_dict['lifter_time'])
        encoder_ts.append(result_dict['encoder_time'])
        render_ts.append(result_dict['render_time'])
        mvs_ts.append(result_dict['mvs_time'])
        dino_ts.append(result_dict['dino_time'])
        head_ts.append(result_dict['head_time'])

        global_iter += 1
        elapsed_epoch += time_e - time_s
        time_per_iter = elapsed_epoch / (i_iter + 1)
        left_evals = cfg.max_epochs // cfg.get('eval_every_epochs', cfg.max_epochs + 1) \
            - epoch // cfg.get('eval_every_epochs', cfg.max_epochs + 1)
        if i_iter % cfg.print_freq == 0 and local_rank == 0:
            time_total = time_e - start_time
            eta_epoch = time_per_iter * (total_iter_per_epoch - i_iter)
            eta_total = time_per_iter * \
                ((cfg.max_epochs - epoch) * total_iter_per_epoch - i_iter) \
                + time_per_iter * left_evals * len(val_loader)
            
            lr = max([p['lr'] for p in optimizer.param_groups])
            # lr = optimizer.param_groups[0]['lr']
            logger.info(
                '[TRAIN] Epoch: %d/%d, Iter: %d/%d, '
                'elapsed: %02dh %02dm %02ds, '
                'ETA: %02dd %02dh %02dm %02ds (%02dh %02dm %02ds)'%(
                epoch + 1, cfg.max_epochs, i_iter, len(train_loader), 
                time_total//3600, (time_total//60)%60, time_total%60,
                eta_total//86400, (eta_total//3600)%24, (eta_total//60)%60, eta_total%60,
                eta_epoch//3600, (eta_epoch//60)%60, eta_epoch%60,)
            )
            logger.info(
                'Loss: %.3f (%.3f), grad_norm: %.3f, lr: %.7f, '
                'time: %.3f (%.3f+%.3f) '%(
                loss.item(), np.mean(loss_list), grad_norm, lr,
                time_e - time_s, data_time_e - data_time_s, time_e - data_time_e,)
            )
            logger.info(
                'Image: %.3f, Lifter: %.3f, Encoder: %.3f, Render: %.3f, MVS: %.3f, DINO: %.3f, Head: %.3f'%(
                sum(image_ts) / len(image_ts),
                sum(lifter_ts) / len(lifter_ts),
                sum(encoder_ts) / len(encoder_ts),
                sum(render_ts) / len(render_ts),
                sum(mvs_ts) / len(mvs_ts),
                sum(dino_ts) / len(dino_ts),
                sum(head_ts) / len(head_ts),)
            )
            detailed_loss = []
            for loss_name, loss_value in loss_dict.items():
                detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
            detailed_loss = ', '.join(detailed_loss)
            logger.info(detailed_loss)
            loss_list = []
            image_ts, lifter_ts, encoder_ts, render_ts, mvs_ts, dino_ts, head_ts = [], [], [], [], [], [], []
        data_time_s = time.time()
        time_s = time.time()

        if cfg.iter_resume:
            if (i_iter + 1) % 50 == 0 and local_rank == 0:
                dict_to_save = {
                    'state_dict': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_iter': global_iter,
                    'last_iter': i_iter + 1,
                }
                iter_num = (cfg.max_epochs - epoch - 1) * total_iter_per_epoch + i_iter
                save_iter_name = os.path.join(os.path.abspath(cfg.work_dir),
                                              f'iter_{iter_num}.pth')
                torch.save(dict_to_save, save_iter_name)
                dst_file = os.path.join(cfg.work_dir, 'latest.pth')
                symlink('iter.pth', dst_file)
                logger.info(f'iter_{iter_num}.pth is saved at {cfg.work_dir}!')
                saved_iters.append(save_iter_name)
                max_keep_iters = cfg.get('max_keep_iters', 1)
                if len(saved_iters) > max_keep_iters:
                    os.remove(saved_iters[0])
                    saved_iters.pop(0)
    
    # save checkpoint
    if local_rank == 0:
        dict_to_save = {
            'state_dict': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'global_iter': global_iter,
        }
        save_epoch_name = os.path.join(os.path.abspath(cfg.work_dir), f'epoch_{epoch + 1}.pth')
        torch.save(dict_to_save, save_epoch_name)
        dst_file = os.path.join(cfg.work_dir, 'latest.pth')
        symlink(f'epoch_{epoch + 1}.pth', dst_file)
        logger.info(f'epoch_{epoch + 1}.pth is saved at {cfg.work_dir}!')

        save_interval = cfg.get('save_interval', 1)
        max_keep_ckpts = cfg.get('max_keep_ckpts', 1)
        save_epoch_list = [save_epoch_name]
        for i in range((epoch + 1) // save_interval, 0, -1):
            if len(save_epoch_list) < max_keep_ckpts + 1:
                save_epoch_list.append(
                    os.path.join(os.path.abspath(cfg.work_dir), f'epoch_{i * save_interval}.pth'))
        save_epoch_list = set(save_epoch_list)
        for ei in range(1, epoch + 1):
            prev_epoch_name = os.path.join(os.path.abspath(cfg.work_dir), f'epoch_{ei}.pth')
            if prev_epoch_name not in save_epoch_list:
                if os.path.exists(prev_epoch_name):
                    os.remove(prev_epoch_name)

    epoch += 1
    first_run = False
    
    # eval
    eval_every_epochs = cfg.get('eval_every_epochs', -1)
    if (eval_every_epochs > 0) and (epoch % eval_every_epochs == 0):
        model.eval()
        os.environ['eval'] = 'true'
        val_loss_list = []

        with torch.no_grad():
            for i_iter_val, data in enumerate(val_loader):
                for k in list(data.keys()):
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].cuda()
                input_imgs = data.pop('img')
                
                with torch.cuda.amp.autocast(amp):
                    result_dict = model(imgs=input_imgs, metas=data)

                    loss_input = {
                        'metas': data,
                        'global_iter': global_iter
                    }
                    for loss_key, loss_val in cfg.loss_input_convertion.items():
                        loss_input.update({
                            loss_key: result_dict[loss_val]})
                    loss, loss_dict = loss_func(loss_input)
                
                if 'final_occ' in result_dict:
                    for idx, pred in enumerate(result_dict['final_occ']):
                        pred_occ = pred
                        gt_occ = result_dict['sampled_label'][idx]
                        occ_mask = result_dict['occ_mask'][idx].flatten()
                        metric._after_step(pred_occ, gt_occ, occ_mask)
                
                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % cfg.print_freq == 0 and local_rank == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch+1, i_iter_val, loss.item(), np.mean(val_loss_list)))
                    detailed_loss = []
                    for loss_name, loss_value in loss_dict.items():
                        detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                    detailed_loss = ', '.join(detailed_loss)
                    logger.info(detailed_loss)
                        
        miou, iou2 = metric._after_epoch()
        logger.info(f'mIoU: {miou}, iou2: {iou2}')
        logger.info('Current val loss is %.3f' % (np.mean(val_loss_list)))
        metric.reset()

    return epoch, global_iter, first_run, saved_iters, saved_epochs

def train_model(
    model,
    cfg,
    distributed,
    logger,
    local_rank=0,
):
    # set distributed
    # logger.info(
    #     f'Params require grad: \
    #         {[n for n, p in model.named_parameters() if p.requires_grad]}'
    #     )
    if distributed:
        if cfg.get('syncBN', True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        model = ddp_model_module(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters
        )
        raw_model = model.module
    else:
        model = model.cuda()
        raw_model = model
    logger.info('done ddp model')

    # get dataloader
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=cfg.iter_resume
    )

    # get optimizer, loss, scheduler
    logger.setLevel('ERROR') # Temporarily suppress mmengine warnings about frozen parameters
    optimizer = build_optim_wrapper(model, cfg.optimizer)

    # Restore original log level
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('multisteplr', False):
        scheduler = MultiStepLRScheduler(
            optimizer,
            **cfg.multisteplr_config
        )
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=cfg.optimizer["optimizer"]["lr"] * cfg.get("min_lr_ratio", 0.1), #1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False
        )
    amp = cfg.get('amp', False)
    if amp:
        scaler = torch.cuda.amp.GradScaler()
        os.environ['amp'] = 'true'
    else:
        scaler = None
        os.environ['amp'] = 'false'
    
    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + cfg.work_dir)

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        try:
            raw_model.load_state_dict(state_dict, strict=False)
        except:
            from misc.checkpoint_util import refine_load_from_sd
            raw_model.load_state_dict(refine_load_from_sd(state_dict), strict=False)

    assert len(cfg.class_names) + 1 == cfg.semantic_dim
    miou_metric = MeanIoU(
        list(range(1, cfg.semantic_dim)),
        len(cfg.class_names) + 1,
        cfg.class_names,
        True,
        len(cfg.class_names) + 1,
        filter_minmax=False
    )
    miou_metric.reset()
    logger.setLevel('INFO')

    first_run = True
    start_time = time.time()
    saved_iters = []
    saved_epochs = []
    while epoch < max_num_epochs:
        epoch, global_iter, first_run, saved_iters, saved_epochs = train_one_epoch(
            model,
            raw_model,
            loss_func,
            train_dataset_loader,
            val_dataset_loader,
            cfg,
            optimizer,
            scheduler,
            logger,
            epoch,
            global_iter,
            first_run,
            start_time,
            amp=amp,
            scaler=scaler,
            local_rank=local_rank,
            last_iter=last_iter,
            metric=miou_metric,
            saved_iters=saved_iters,
            saved_epochs=saved_epochs,
        )