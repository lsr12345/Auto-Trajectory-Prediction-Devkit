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

from model.models import LaneGcn
from tools.loss.loss import LaneGcnLoss

from utils.common import reduce_mean
from utils.standard_tools import togpu, recursiveToTensor

class LaneGcn_Model():
    def __init__(self, config, n_actor=128, n_map=128, num_mods=6, num_pred_points=30, optimizer_name='adamW', scheduler_name='Cosine', amp_training=False, act='relu', supervision_angular=False, add_theta_info=False):
        self.config = config
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
        self.num_pred_points = int(config.get('pred_points', num_pred_points))
        self.num_mods = int(config.get('num_mods', num_mods))
        self.n_actor = int(config.get('n_actor', n_actor))
        self.n_map = int(config.get('n_map', n_map))
        self.supervision_angular = config.get('supervision_angular', supervision_angular)

    def get_model(self):
        model = LaneGcn(n_actor=self.n_actor, n_map=self.n_map, num_mods=self.num_mods, num_pred_points=self.num_pred_points, 
                                            supervision_angular=self.supervision_angular, train_model=True)
        return model

    def get_inference_model(self):
        model = LaneGcn(n_actor=self.n_actor, n_map=self.n_map, num_mods=self.num_mods, num_pred_points=self.num_pred_points, 
                                            supervision_angular=self.supervision_angular, train_model=False)
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
                            cls_th=2, cls_ignore=0.2, mgn=0.2, cls_coef=1.0, reg_coef=1.0, ang_coef=1.0, calculate_angular_loss=self.supervision_angular)
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1,  loss_criteria=True):
        eval_nums = len(data_loader)
        model.eval()
        eval_loss = 0.
        metrics = dict()

        if distributed:
            loss_list = []
            ade1_list = []
            fde1_list = []
            ade_list = []
            fde_list = []

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
                    ade1, fde1, ade, fde, min_idcs = self.calculate_metrics(metrics)
                    torch.distributed.barrier()
                    loss= reduce_mean(loss_out["loss"], nprocs)
                    ade1= reduce_mean(ade1, nprocs)
                    fde1= reduce_mean(fde1, nprocs)
                    ade= reduce_mean(ade, nprocs)
                    fde= reduce_mean(fde, nprocs)
                    metrics = dict()
                    loss_list.append(loss)
                    ade1_list.append(ade1)
                    fde1_list.append(fde1)
                    ade_list.append(ade)
                    fde_list.append(fde)
                else:
                    eval_loss += loss_out["loss"].item()
                    
        if not distributed:
            ade1, fde1, ade, fde, min_idcs = self.calculate_metrics(metrics)
            if is_main_process:
                logger.info("** ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"% (ade1, fde1, ade, fde))
            return eval_loss/eval_nums
        else:
            ade1= torch.mean(torch.tensor(ade1_list))
            fde1= torch.mean(torch.tensor(fde1_list))
            ade= torch.mean(torch.tensor(ade_list))
            fde= torch.mean(torch.tensor(fde_list))
            if is_main_process:
                logger.info("** ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"% (ade1, fde1, ade, fde))

            loss = torch.mean(torch.tensor(loss_list))
            return loss


    def postprocess(self, y_pr,  y_gt, loss_out, metrics):

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

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg
        return metrics


    def calculate_metrics(self, metrics):
        preds = torch.cat(metrics["preds"], 0)
        gt_preds =  torch.cat(metrics["gt_preds"], 0)
        has_preds =  torch.cat(metrics["has_preds"], 0)

        err = torch.sqrt(((preds - torch.unsqueeze(gt_preds, 1)) ** 2).sum(3))

        ade1 = err[:, 0].mean()
        fde1 = err[:, 0, -1].mean()

        min_idcs = err[:, :, -1].argmin(1)
        row_idcs = np.arange(min_idcs.size()[0]).astype(np.int64)
        err = err[row_idcs, min_idcs]
        ade = err.mean()
        fde = err[:, -1].mean()
        return ade1, fde1, ade, fde, min_idcs

    def preprocess(self, data):
        mod = 6
        input_keys = ["actors", "graph"]
        concat_batch = ["feats", "graph_feats", "graph_turn", "graph_control", "graph_intersect"]
        nonconcat_bath = ["actor_ctrs", "graph_ctrs"]
        common_keys = ["rot", "orig", "theta"]

        batch_data = [data]
        batch_data = recursiveToTensor(batch_data)
        batch_size = len(batch_data)
        num_actors = [(x["actors"]["feats"].shape[0]) for x in batch_data]

        actor_idcs = []
        actor_count = 0

        node_idcs = []
        node_count = 0
        node_counts = []

        if self.add_theta_info:
            for i in range(batch_size):
                theta = batch_data[i]["theta"].clone().detach().unsqueeze(-1).unsqueeze(-1).float()
                batch_data[i]["actors"]["feats"] = torch.concat([batch_data[i]["actors"]["feats"], theta.repeat(batch_data[i]["actors"]["feats"].size()[0],batch_data[i]["actors"]["feats"].size()[1], 1)], dim=-1)
                theta =  batch_data[i]["theta"].clone().detach().unsqueeze(-1).float()
                batch_data[i]["graph"]["graph_feats"] = torch.concat([batch_data[i]["graph"]["graph_feats"], theta.repeat(batch_data[i]["graph"]["graph_feats"].size()[0], 1)], dim=-1)

        for i in range(batch_size):
            a_idcs = torch.arange(actor_count, actor_count + num_actors[i])

            actor_idcs.append(a_idcs)
            actor_count += num_actors[i]

            node_counts.append(node_count)
            n_idcs = torch.arange(node_count, node_count + batch_data[i]["graph"]["num_nodes"])

            node_idcs.append(n_idcs)
            node_count = node_count + batch_data[i]["graph"]["num_nodes"]

        return_input_batch = dict()

        for key in common_keys:
            return_input_batch[key] = []
        for key in common_keys:
            return_input_batch[key] = [x[key].float() for x in batch_data]


        for key in input_keys:
            return_input_batch[key] = dict()
            for ke in batch_data[0][key]:
                if ke != "num_nodes":
                    return_input_batch[key][ke] = dict()

                if ke in concat_batch:
                    return_input_batch[key][ke] = torch.cat([x[key][ke] for x in batch_data], 0)

                elif ke in nonconcat_bath:

                    return_input_batch[key][ke] = [x[key][ke] for x in batch_data]
                
                elif ke in ["graph_pre", "graph_suc"]:
                    return_input_batch[key][ke] = []
                    for ii in range(mod):
                        return_input_batch[key][ke].append(dict())
                        for k in ['v', 'u']:
                            return_input_batch[key][ke][ii][k] = torch.cat([batch_data[iii][key][ke][ii][k].to(torch.int32) + node_counts[iii]for iii in range(batch_size)], 0)              
                elif ke in ["graph_left", "graph_right"]:
                    for k in ['v', 'u']:
                        temp = [batch_data[i][key][ke][k].to(torch.int32) + node_counts[i] for i in range(batch_size)]
                        temp = [
                            x if x.dim() > 0 else batch_data[0]["graph"]["graph_pre"]["u"].new().resize_(0)  
                            for x in temp
                        ]
                        return_input_batch[key][ke][k] = torch.cat(temp)
                    
                elif ke in ["num_nodes", "lane_idcs"]:
                    continue

                else:
                    raise NotImplementedError('check data key:  {} '.format(ke))

        return_input_batch["actors"]["actor_idcs"] = actor_idcs
        return_input_batch["graph"]["graph_idcs"] = node_idcs

        return return_input_batch

