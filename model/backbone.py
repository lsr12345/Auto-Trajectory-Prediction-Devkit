'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 主干网络

example:

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from tools.nninit import common_init

from model.utils.lanegcn_utils import MapNet, ActorNet
from model.utils.vectornet_utils import PolyGraph, SACat_Layer

class LaneGcnBackbone(nn.Module):

    def __init__(self, n_actor=128, n_map=128, add_theta_info=False):
        super(LaneGcnBackbone, self).__init__()
        self.actor_net = ActorNet(n_actor, add_theta_info=add_theta_info)
        self.map_net = MapNet(n_map, add_theta_info=add_theta_info)

        self.apply(self._init_weights)

    def forward(self, actors, graph_ctrs, graph_feats, graph_pre, graph_suc, graph_left, graph_right):
        actors = actors.transpose(1, 2)
        actors = self.actor_net(actors)
        nodes = self.map_net(graph_ctrs, graph_feats, graph_pre, graph_suc, graph_left, graph_right)
        return actors, nodes

    def _init_weights(self, m):
        common_init(m)


class VectornetBackBone(nn.Module):

    def __init__(self, hidden_size):
        super(VectornetBackBone, self).__init__()
        self.objects_layers = PolyGraph(hidden_size=hidden_size, depth=3,  norm = "LN", act = "relu")
        self.lanes_layers = PolyGraph(hidden_size=hidden_size, depth=3,  norm = "LN", act = "relu")
        self.global_layers = SACat_Layer(hidden_size=hidden_size)

        self.apply(self._init_weights)

    def forward(self, matrix_objects_batch_list, matrix_lanes_vectors_batch_list,  device=None):
        batch_size = len(matrix_objects_batch_list)
        if device is None:
            device = matrix_objects_batch_list[0].device
        global_states_list = []
        for i in range(batch_size):
            objects_states = self.objects_layers(matrix_objects_batch_list[i])
            lanes_states = self.lanes_layers(matrix_lanes_vectors_batch_list[i])
            global_states = torch.cat((objects_states, lanes_states), dim=0).unsqueeze(0)
            global_states = torch.cat((global_states, self.global_layers(global_states)), dim=-1)
            global_states_list.append(global_states)
        return global_states_list

    def _init_weights(self, m):
        common_init(m)