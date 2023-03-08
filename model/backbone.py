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
