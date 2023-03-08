'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: neck net

example:

'''

import torch.nn as nn
import torch.nn.functional as F

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from tools.nninit import  common_init

from model.utils.lanegcn_utils import FilterDistance, A2A, A2M, M2A, M2M

import torch.nn as nn
import torch.nn.functional as F

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from tools.nninit import  common_init

from model.utils.lanegcn_utils import FilterDistance, A2A, A2M, M2A, M2M

class LaneGcnNeck(nn.Module):
    def __init__(self, n_actor=128, n_map=128):
        super(LaneGcnNeck, self).__init__()
        self.a2m = A2M(n_actor=n_actor, n_map=n_map)
        self.a2m_dist = FilterDistance(dist_th=7.0)

        self.m2m = M2M(n_map=n_map)

        self.m2a = M2A(n_map=n_map, n_actor=n_actor)
        self.m2a_dist = FilterDistance(dist_th=6.0)

        self.a2a = A2A(n_actor=n_actor)
        self.a2a_dist = FilterDistance(dist_th=100.0)

        self.apply(self._init_weights)

    def forward(self, actors, nodes, actor_idcs, actor_ctrs, graph_idcs, graph_ctrs, graph_turn, graph_control, graph_intersect,
                            graph_pre, graph_suc, graph_left, graph_right):

        distance_a2m, hi_a2m, wi_a2m = self.a2m_dist(graph_idcs, graph_ctrs, actor_idcs, actor_ctrs)

        nodes = self.a2m(nodes,  graph_turn, graph_control, graph_intersect, actors, distance_a2m, hi_a2m, wi_a2m)

        nodes = self.m2m(nodes, graph_pre, graph_suc, graph_left, graph_right)

        distance_m2a, hi_m2a, wi_m2a = self.m2a_dist(actor_idcs, actor_ctrs, graph_idcs, graph_ctrs)
        actors = self.m2a(actors, nodes, distance_m2a, hi_m2a, wi_m2a)

        distance_a2a, hi_a2a, wi_a2a = self.a2a_dist(actor_idcs, actor_ctrs, actor_idcs, actor_ctrs)
        actors = self.a2a(actors, distance_a2a, hi_a2a, wi_a2a)

        return actors

    def _init_weights(self, m):
        common_init(m)

