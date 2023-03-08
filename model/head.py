'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: head net

example:

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.utils.ops import MLP
from tools.nninit import common_init

from model.utils.lanegcn_utils import AttDest

class LaneGcnHead(nn.Module):
    def __init__(self, n_actor=128, num_mods=6, num_pred_points=30, supervision_angular=False, train_model=True):
        super(LaneGcnHead, self).__init__()
        norm = "GN"
        act = "relu"
        group_num = 1
        self.n_actor = n_actor
        self.num_mods = num_mods
        self.num_pred_points = num_pred_points

        self.supervision_angular = supervision_angular
        self.train_model = train_model

        pred = []
        for _ in range(self.num_mods):
            pred.append(
                nn.Sequential(
                    MLP(n_actor, n_actor,  bias=False, act=act, norm=norm, group_num=group_num, res_type=True),
                    nn.Linear(n_actor, 2 * self.num_pred_points)
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            MLP(n_actor, n_actor,  bias=False, act=act, norm=norm, group_num=group_num, res_type=True),
            nn.Linear(n_actor, 1)
        )

        if self.supervision_angular:
            angular_pred = []
            for _ in range(self.num_mods):
                angular_pred.append(
                    nn.Sequential(
                        MLP(n_actor, n_actor,  bias=False, act=act, norm=norm, group_num=group_num, res_type=False),
                        nn.Linear(n_actor, 1)
                    )
                )
            self.angular_pred = nn.ModuleList(angular_pred)


        self.apply(self._init_weights)

    def forward(self, actors, actor_idcs, actor_ctrs):
        preds = []
        if self.supervision_angular:
            angulars = []
        for i in range(self.num_mods):
            preds.append(self.pred[i](actors))
            if self.supervision_angular:
                angulars.append(self.angular_pred[i](actors))

        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        if self.supervision_angular:
            angulars = torch.cat([angular for angular in angulars], 1)

        for ii in range(len(actor_idcs)):
            idcs = actor_idcs[ii]
            ctrs = actor_ctrs[ii].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.num_mods)

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        if self.supervision_angular:
            angulars =  angulars[row_idcs, sort_idcs].view(cls.size(0), cls.size(1))


        if self.train_model:
            out = dict()
            out["cls"], out["reg"] = [], []
            out["angular"] = []
            for i in range(len(actor_idcs)):
                idcs = actor_idcs[i]
                out["cls"].append(cls[idcs])
                out["reg"].append(reg[idcs])
                if self.supervision_angular:
                    out["angular"].append(angulars[idcs])
            return out
        else:
            if self.supervision_angular:
                return reg, angulars
            else:
                return cls[idcs], reg[idcs]
        

    def _init_weights(self, m):
        common_init(m)

class CommonTrajPredicter(nn.Module):
    def __init__(self, n_actor=128, num_mods=6, num_pred_points=30, train_model=True):
        super(CommonTrajPredicter, self).__init__()
        norm = "LN"
        act = "relu"
        self.n_actor = n_actor
        self.num_mods = num_mods
        self.num_pred_points = num_pred_points

        self.train_model = train_model

        self.pred =  nn.Sequential(
                    MLP(n_actor*2, n_actor,  bias=True, act=act, norm=norm, res_type=True),
                    nn.Linear(n_actor, 2 * self.num_pred_points)
                )

        self.apply(self._init_weights)

    def forward(self, actors):
        if self.train_model:
            out = dict()
            out["reg"] = []
            batch_size = len(actors)
            for b in range(batch_size):
                preds = self.pred(actors[b])
                reg = preds.view(preds.size(1), 30, 2)
                out["reg"].append(reg)
            return out

        else:
            return 0
        

    def _init_weights(self, m):
        common_init(m)
