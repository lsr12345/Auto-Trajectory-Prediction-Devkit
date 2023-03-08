'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 算法 NET

example:

'''

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from typing import  Dict

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.backbone import LaneGcnBackbone
from model.neck import LaneGcnNeck
from model.head import LaneGcnHead


class LaneGcn(nn.Module):
    def __init__(self, n_actor=128, n_map=128, num_mods=6, num_pred_points=30,
                        supervision_angular=False, train_model=True, add_theta_info=False):
        super().__init__()
        self.train_model = train_model

        self.backbone = LaneGcnBackbone(n_actor=n_actor, n_map=n_map)
        self.neck = LaneGcnNeck(n_actor=n_actor, n_map=n_map, )
        self.head = LaneGcnHead(n_actor=n_actor, num_mods=num_mods, num_pred_points=num_pred_points, supervision_angular=supervision_angular, train_model=self.train_model)

    def forward(self, x:Dict):
        actors = x["actors"]["feats"]
        actor_idcs = x["actors"]["actor_idcs"]
        actor_ctrs = x["actors"]["actor_ctrs"]

        graph_ctrs = x["graph"]["graph_ctrs"]
        graph_feats = x["graph"]["graph_feats"]
        graph_idcs = x["graph"]["graph_idcs"]

        graph_turn = x["graph"]["graph_turn"]
        graph_control = x["graph"]["graph_control"]
        graph_intersect = x["graph"]["graph_intersect"]


        graph_pre = x["graph"]["graph_pre"]
        graph_suc = x["graph"]["graph_suc"]
        graph_left = x["graph"]["graph_left"]
        graph_right = x["graph"]["graph_right"]
        
        if self.train_model:
            rot, orig = x["rot"], x["orig"]

        actors, nodes = self.backbone(actors, graph_ctrs, graph_feats, graph_pre, graph_suc, graph_left, graph_right)

        actors = self.neck(actors, nodes, actor_idcs, actor_ctrs, graph_idcs, graph_ctrs, graph_turn, graph_control,
                                            graph_intersect, graph_pre, graph_suc, graph_left, graph_right)
        
        outputs = self.head(actors, actor_idcs, actor_ctrs)

        if self.train_model:
            for i in range(len(outputs["reg"])):
                outputs["reg"][i] = torch.matmul(outputs["reg"][i], rot[i]) + orig[i].view(
                    1, 1, 1, -1
                )

        return outputs

    def load_pretrained_model(self, filename):
        state_dict_names = ["state_dict", "model"]
        match_tensor = 0
        pretrained_state_dict = torch.load(filename,  map_location="cpu")
        for name in state_dict_names:
            if name in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict[name]
                break
        self_state_dict = self.backbone.state_dict()
        for k, v in pretrained_state_dict.items():
            if k in self_state_dict:
                self_state_dict.update({k: v})
                match_tensor += 1
        print("Pretrained Model inital tensors: {}/{}".format(match_tensor, len(pretrained_state_dict)))
        self.backbone.load_state_dict(self_state_dict, strict=False)
