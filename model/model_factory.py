'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: Model 类

example:

'''

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from loguru import logger
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.models import LaneGcn, VectorNet
from tools.loss.loss import LaneGcnLoss, VectorNetLoss

from utils.common import reduce_mean
from utils.standard_tools import togpu

class LaneGcn_Model():
    def __init__(self, config, n_actor=128, n_map=128, num_mods=6, num_pred_points=30, optimizer_name='adamW',
                 scheduler_name='Cosine', amp_training=False, act='relu', supervision_angular=False, add_theta_info=False):
        self.config = config
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
        self.num_pred_points = int(config.get('pred_points', num_pred_points))
        self.num_mods = int(config.get('num_mods', num_mods))
        self.n_actor = int(config.get('n_actor', n_actor))
        self.n_map = int(config.get('n_map', n_map))
        self.supervision_angular = config.get('supervision_angular', supervision_angular)

    def get_model(self):
        model = LaneGcn(n_actor=self.n_actor, n_map=self.n_map, num_mods=self.num_mods,
                        num_pred_points=self.num_pred_points, supervision_angular=self.supervision_angular, train_model=True)
        return model

    def get_inference_model(self):
        model = LaneGcn(n_actor=self.n_actor, n_map=self.n_map, num_mods=self.num_mods,
                        num_pred_points=self.num_pred_points, supervision_angular=self.supervision_angular, train_model=False)
        return model
    
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)

        else:
            return None
    
    def get_loss_func(self):
        return LaneGcnLoss(num_mods=self.num_mods, num_pred_points=self.num_pred_points, 
                            cls_th=2, cls_ignore=0.2, mgn=0.2, cls_coef=1.0, reg_coef=1.0,
                            ang_coef=1.0, calculate_angular_loss=self.supervision_angular)
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1, loss_criteria=True):
        eval_nums = len(data_loader)
        model.eval()
        eval_loss = 0.
        metrics = dict()

        if distributed:
            loss_list = []
            ade1_list = []
            fde1_list = []
            mr1_list = []
            ade_list = []
            fde_list = []
            mr6_list = []

        with torch.no_grad():
            for inps, targets in data_loader:

                if not isinstance(inps ,dict):
                    inps = inps.cuda(non_blocking=True)
                    inps.requires_grad = False
                elif isinstance(inps ,dict):
                    inps = togpu(inps)

                if not isinstance(targets ,dict) and not isinstance(targets, list):
                    targets = targets.cuda(non_blocking=True)
                    targets.requires_grad = False
                elif isinstance(targets ,dict):
                    targets = togpu(targets)

                outputs = model(inps)
                loss_out = loss_fn(outputs, targets, training=False)
                metrics = self.postprocess(outputs, targets, loss_out, metrics)
            
                if distributed:
                    ade1, fde1, ade, fde, min_idcs, mr1, mr6 = self.calculate_metrics(metrics)
                    torch.distributed.barrier()
                    loss= reduce_mean(loss_out["loss"], nprocs)
                    ade1= reduce_mean(ade1, nprocs)
                    fde1= reduce_mean(fde1, nprocs)
                    mr1 = reduce_mean(mr1, nprocs)
                    ade= reduce_mean(ade, nprocs)
                    fde= reduce_mean(fde, nprocs)
                    mr6 = reduce_mean(mr6, nprocs)
                    metrics = dict()
                    loss_list.append(loss)
                    ade1_list.append(ade1)
                    fde1_list.append(fde1)
                    mr1_list.append(mr1)
                    ade_list.append(ade)
                    fde_list.append(fde)
                    mr6_list.append(mr6)
                else:
                    eval_loss += loss_out["loss"].item()
                    
        if not distributed:
            ade1, fde1, ade, fde, min_idcs, mr1, mr6 = self.calculate_metrics(metrics)
            if is_main_process:
                logger.info("** ade1 %2.4f, fde1 %2.4f, mr1 %2.4f, ade %2.4f, fde %2.4f, mr %2.4f"% (ade1, fde1, mr1, ade, fde, mr6))
            return eval_loss/eval_nums
        else:
            ade1= torch.mean(torch.tensor(ade1_list))
            fde1= torch.mean(torch.tensor(fde1_list))
            mr1= torch.mean(torch.tensor(mr1_list))
            ade= torch.mean(torch.tensor(ade_list))
            fde= torch.mean(torch.tensor(fde_list))
            mr6= torch.mean(torch.tensor(mr6_list))
            if is_main_process:
                logger.info("** ade1 %2.4f, fde1 %2.4f, mr1 %2.4f, ade %2.4f, fde %2.4f, mr %2.4f"% (ade1, fde1, mr1, ade, fde, mr6))

            loss = torch.mean(torch.tensor(loss_list))
            return loss

    def postprocess(self, y_pr, y_gt, loss_out, metrics):
        post_out = dict()

        post_out["preds"] = [x[0:1].detach() for x in y_pr["reg"]]
        post_out["gt_preds"] = [x[0:1] for x in y_gt["gt_preds"]]
        post_out["has_preds"] = [x[0:1] for x in y_gt["has_preds"]]

        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]

        return metrics

    def calculate_metrics(self, metrics):
        preds = torch.cat(metrics["preds"], 0)
        gt_preds =  torch.cat(metrics["gt_preds"], 0)
        has_preds =  torch.cat(metrics["has_preds"], 0)

        err = torch.sqrt(((preds - torch.unsqueeze(gt_preds, 1)) ** 2).sum(3))

        ade1 = err[:, 0].mean()
        fde1 = err[:, 0, -1].mean()

        mr1 = torch.sum(err[:, 0, -1] > 2) / err.size(0)

        min_idcs = err[:, :, -1].argmin(1)
        row_idcs = np.arange(min_idcs.size()[0]).astype(np.int64)
        err = err[row_idcs, min_idcs]
        ade = err.mean()
        fde = err[:, -1].mean()
        mr6 = torch.sum(err[:, -1] > 2) / err.size(0)
        return ade1, fde1, ade, fde, min_idcs, mr1, mr6

class VectorNet_Model():
    def __init__(self, config, hidden_size=128, num_mods=6, future_frame_num=30,
                 optimizer_name='adamW', scheduler_name='Cosine', amp_training=False):
        self.config = config
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
        self.future_frame_num = int(config.get('num_pred_points', future_frame_num))
        self.num_mods = int(config.get('num_mods', num_mods))
        self.hidden_size = config.get('hidden_size', hidden_size)

    def get_model(self):
        model = VectorNet(hidden_size=self.hidden_size, num_mods=self.num_mods, future_frame_num=self.future_frame_num)
        return model

    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)

        else:
            return None
    
    def get_loss_func(self):
        return VectorNetLoss()
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1,  loss_criteria=True):
        eval_nums = len(data_loader)
        model.eval()
        eval_loss = 0.
        ADES = []
        FDES = []
        MRS = []

        with torch.no_grad():
            for inps, targets in data_loader:

                if not isinstance(inps ,dict):
                    inps = inps.cuda(non_blocking=True)
                    inps.requires_grad = False
                elif isinstance(inps ,dict):
                    inps = togpu(inps)

                if not isinstance(targets ,dict) and not isinstance(targets, list):
                    targets = targets.cuda(non_blocking=True)
                    targets.requires_grad = False
                elif isinstance(targets ,dict):
                    targets = togpu(targets)

                outputs = model(inps)
                loss_out, DE = loss_fn(outputs, targets, training=False)
                ade = torch.mean(DE)
                fde = torch.mean(DE[:, -1])
                mr = torch.sum(DE[:, -1] > 2) / DE.size(0)
                ADES.append(ade.item())
                FDES.append(fde.item())
                MRS.append(mr.item())
                eval_loss += loss_out["loss"].item()

            if is_main_process:
                logger.info("** ADE_1: %2.4f, FDE_1: %2.4f, MR: %2.4f"% (np.array(ADES).mean(), np.array(FDES).mean(), np.array(MRS).mean()))
            return eval_loss/eval_nums

