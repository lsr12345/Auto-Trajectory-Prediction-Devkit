'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: loss

example:

'''

import torch
import torch.nn as nn
from typing import  Dict

import math

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))


class LaneGcnLoss(nn.Module):

    def __init__(self, num_mods=6, num_pred_points=30, cls_th=2, cls_ignore=0.2, mgn=0.2, cls_coef=1.0, reg_coef=1.0,  ang_coef=1.0,
                calculate_angular_loss=False):
        super(LaneGcnLoss, self).__init__()
        self.num_mods = num_mods
        self.num_pred_points = num_pred_points
        self.cls_th = cls_th
        self.cls_ignore = cls_ignore
        self.mgn = mgn
        self.cls_coef = cls_coef
        self.reg_coef = reg_coef
        self.ang_coef = ang_coef
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

        self.calculate_angular_loss = calculate_angular_loss
        if self.calculate_angular_loss:
            self.angular_loss = nn.SmoothL1Loss(reduction="mean")

    def forward(self, y_pr: Dict, y_gt: Dict, training=True):
        cls, reg = y_pr["cls"], y_pr["reg"]
        gt_preds, has_preds = y_gt["gt_preds"], y_gt["has_preds"]

        if self.calculate_angular_loss:
            pr_angular = y_pr["angular"]
            pr_angular = torch.cat([x for x in pr_angular], 0)
            rot_theta =  y_gt["theta"]
            rot_thetas = []
            for b in range(len(gt_preds)):
                rot_thetas.extend(gt_preds[b].size()[0] * [rot_theta[b]])
        gt_preds = torch.cat([x for x in gt_preds], 0)
        gt_preds_ = gt_preds.clone()
        has_preds = torch.cat([x for x in has_preds], 0)
        has_preds_ = has_preds.clone()

        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0
        loss_out["angular_loss"] = zero.clone()

        last = has_preds.float() + 0.1 * torch.arange(self.num_pred_points).float().to(has_preds.device) / float(self.num_pred_points)

        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]

        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        
        dist = []
        for j in range(self.num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.cls_th).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.cls_ignore
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.mgn
        loss_out["cls_loss"] += self.cls_coef  * (
            self.mgn * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()
        reg = reg[row_idcs, min_idcs]
        loss_out["reg_loss"] += self.reg_coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()

        if not self.calculate_angular_loss:
            loss_out["loss"] = loss_out["cls_loss"] / (
                loss_out["num_cls"] + 1e-10
            ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        
        else:
            pr_angular = pr_angular[row_idcs, min_idcs]
            loss_out["angular_loss"] += self.ang_coef * self.angular_loss_func(pr_angular, gt_preds_, has_preds_, rot_thetas, row_idcs)
            loss_out["loss"] = loss_out["cls_loss"] / (
                loss_out["num_cls"] + 1e-10
            ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10) + loss_out["angular_loss"]
        if training:
            return  loss_out["loss"]
        else:
            return loss_out
        
    def angular_loss_func(self, pre_angular, gt_points, has_preds, rot_thetas, row_idcs):
        gt_points = gt_points[row_idcs]
        thetas = []
        valid_index = []
        for i, points in enumerate(gt_points):
            points = points[has_preds[i]]
            if points.size()[0] >=30:
                valid_index.append(i)
                pre = points[-5:-1] - points[-1]
                pre_thetas = [(math.pi - torch.atan2(p[1], p[0])) for p in pre]
                pre_thetas = [int(i * 180/math.pi) for i in pre_thetas]
                theta = self.calc_angles_mean(pre_thetas)
                theta = theta / 180 * math.pi
                thetas.append((theta-rot_thetas[i])/(math.pi*2))

        thetas = torch.tensor(thetas).to(pre_angular.device)

        loss = self.angular_loss(pre_angular[valid_index], thetas)

        return loss

    def calc_angles_mean(self, angle_list):
        m_sin, m_cos = self.calc_mean_sin_cos(angle_list)
    
        mean_angle = math.atan(m_sin/(m_cos+1e-9))*(180/math.pi)
        if (m_cos >= 0 ):
            mean_angle = mean_angle
        elif (m_cos < 0):
            mean_angle = mean_angle + 180
        if mean_angle < 0 :
            mean_angle = 360 + mean_angle
        return mean_angle

    def calc_mean_sin_cos(self, angle_list):
        m_sin = 0
        m_cos = 0
        for data_elmt in angle_list:
            m_sin += math.sin(math.radians(data_elmt))
            m_cos += math.cos(math.radians(data_elmt))
        m_sin /= len(angle_list)
        m_cos /= len(angle_list)
    
        return m_sin, m_cos
