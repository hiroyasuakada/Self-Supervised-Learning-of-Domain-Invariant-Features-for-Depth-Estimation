import time
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from dataloader.data_loader_val import dataloader_val
from model.models import create_model
from util.util import tensor2im, RunningAverage, RunningAverageDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import os
import cv2

import torch.multiprocessing as mp
import torch.distributed as dist

try:
    import torch.cuda.amp as amp  
except ModuleNotFoundError as e:
    pass

from util.process_data import compute_errors
from util.dist_util import reduce_loss_dict, get_rank
import uuid


def validate(opt, model, val_dataset):
    metrics = RunningAverageDict()

    model.netTask.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataset), total=len(val_dataset), desc="Val: "):
            img_target_val = data['img_target_val'].cuda(opt.gpu_ids[0])
            lab_target_val = data['lab_target_val']
            lab_size = data['lab_size']

            outputs = model.netTask.module(img_target_val)
            # outputs, _ = model.netTask.module(img_target_val)
            depth = outputs[-1] # [0:1] scale
            depth = depth.squeeze(0).cpu().float().numpy()
            depth = np.transpose(depth, (1, 2, 0)) * opt.normalize_depth # [0.0:10.0] scale for nyu, [0.0:80.0] scale for kitti
            depth = depth[:, :, 0] 

            if opt.dataset == 'kitti':
                depth = cv2.resize(depth,(lab_size[1][0], lab_size[0][0]),interpolation=cv2.INTER_LINEAR)            

            # compute metrics, see evaluate.py for more info
            ground_depth = lab_target_val.squeeze(0).cpu().float().numpy() # delete B
            predicted_depth = depth

            height, width = ground_depth.shape
            _height, _width = predicted_depth.shape

            if not height == _height:
                predicted_depth = cv2.resize(predicted_depth,(width,height),interpolation=cv2.INTER_LINEAR)

            mask = np.logical_and(ground_depth > opt.min_depth, ground_depth < opt.max_depth)

            # crop used by Garg ECCV16
            if opt.garg_crop and opt.dataset == 'kitti':
                crop = np.array([0.40810811 * height,  0.99189189 * height,
                                        0.03594771 * width,   0.96405229 * width]).astype(np.int32)

            # crop we found by trail and error to reproduce Eigen NIPS14 results
            elif opt.eigen_crop and opt.dataset == 'nyu':
                crop = np.array([0.3324324 * height,  0.91351351 * height,
                                        0.0359477 * width,   0.96405229 * width]).astype(np.int32)

            # else:
            #     raise ValueError('garg_crop for kitti, eigen_crop for nyu')

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            ground_depth = ground_depth[mask]
            predicted_depth = predicted_depth[mask]

            predicted_depth[predicted_depth < opt.min_depth] = opt.min_depth
            predicted_depth[predicted_depth > opt.max_depth] = opt.max_depth

            abs_rel, sq_rel, rmse, rmse_log, log_10, a1, a2, a3 = compute_errors(ground_depth,predicted_depth)
            metrics.update(
                dict(abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse, rmse_log=rmse_log, log_10=log_10, a1=a1, a2=a2, a3=a3)
            )
    
    model.netTask.train()

    return metrics.get_value()


    def print_current_errors(self, epoch, i, errors):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)


if __name__ == '__main__':

    opt = TrainOptions().parse()

    # note that depth maps are scaled to [0:1] in dataloader.py
    if opt.dataset == 'nyu':
        opt.normalize_depth = 10.0
        opt.max_depth = 10.0 
        opt.min_depth = 1e-3 
    elif opt.dataset == 'kitti':
        opt.normalize_depth = 80.0
        opt.max_depth = 80.0 
        opt.min_depth = 1e-3 

    train_dataset = dataloader(opt)
    val_dataset = None
    if opt.no_online_eval is not True:
        if opt.model == 'full_fine':
            val_dataset = dataloader_val(opt, mode='val')
    dataset_size = len(train_dataset) * opt.batch_size
    print('training images = %d' % dataset_size)

    model = create_model(opt)

    total_steps=0
    best_loss_abs_rel = np.inf
    best_metrics = None
    for epoch in range(opt.epoch_count, opt.niter+opt.niter_decay+1):
        epoch_start_time = time.time()
        epoch_iter = 0

        # training
        for i, data in tqdm(enumerate(train_dataset), total=len(train_dataset), desc=f"Epoch: {epoch}"):

            iter_start_time = time.time()
            total_steps += 1
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(i)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                print_current_errors(epoch, epoch_iter, errors)

            if total_steps % opt.save_latest_freq == 0:
                model.save_networks('latest')

            if (total_steps % opt.val_freq == 0) and (val_dataset is not None):
                metrics = validate(opt, model, val_dataset)

                if metrics['abs_rel'] < best_loss_abs_rel:
                    best_loss_abs_rel = metrics['abs_rel']
                    model.save_networks('best')

                    best_metrics = metrics

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)

        model.update_learning_rate()

        print('dir name: {}'.format(opt.experiment_name)) 
    
    # print best metrics
    if val_dataset is not None:
        print('===== Best Results =====')
        print('abs_rel  : {}'.format(best_metrics['abs_rel']))
        print('sq_rel   : {}'.format(best_metrics['sq_rel']))
        print('rmse     : {}'.format(best_metrics['rmse']))
        print('rmse_log : {}'.format(best_metrics['rmse_log']))
        print('log_10   : {}'.format(best_metrics['log_10'])) # only for evaluation on NYUv2
        print('a1       : {}'.format(best_metrics['a1']))
        print('a2       : {}'.format(best_metrics['a2']))
        print('a3       : {}'.format(best_metrics['a3']))
