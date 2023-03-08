'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from fractions import gcd

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.utils.ops import CBA, MLP
from tools.nninit import common_init


class ActorNet(nn.Module):
    def __init__(self, n_actor=128, add_theta_info=False):
        super(ActorNet, self).__init__()
        norm = "GN"
        act = "relu"
        group_num = 1
        num_in = 3 if not add_theta_info else 4
        num_outs = [32, 64, 128]

        groups = []
        lateral = []

        for i in range(len(num_outs)):
            group = []
            if i == 0:
                group.append(CBA(num_in, num_outs[i], 3, 1, act=act, use_bn=True, norm=norm, group_num=group_num, conv='conv1d', res_type=True))
            else:
                group.append(CBA(num_in, num_outs[i], 3, 2, act=act, use_bn=True, norm=norm, group_num=group_num, conv='conv1d', res_type=True))

            group.append(CBA(num_outs[i], num_outs[i], 3, 1, act=act, use_bn=True, norm=norm, group_num=group_num, conv='conv1d', res_type=True))

            lateral.append(CBA(num_outs[i], n_actor, 3, 1, act=None, use_bn=True, norm=norm, group_num=group_num, conv='conv1d'))
            groups.append(nn.Sequential(*group))

            num_in = num_outs[i]

        self.groups = nn.ModuleList(groups)
        self.lateral = nn.ModuleList(lateral)

        self.output = CBA(n_actor, n_actor, 3, 1, act=act, use_bn=True,norm=norm, group_num=group_num, conv='conv1d', res_type=True)


        self.apply(self._init_weights)

    def forward(self, x):
        outputs = []
        for i in range(len(self.groups)):
            x = self.groups[i](x)
            outputs.append(x)

        x = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)

            x += self.lateral[i](outputs[i])

        x = self.output(x)[:, :, -1]

        return x

    def _init_weights(self, m):
        common_init(m)


class MapNet(nn.Module):
    def __init__(self, n_map=128, add_theta_info=False):
        super(MapNet, self).__init__()
        norm = "GN"
        group_num = 1
        self.num_blocks = 4
        self.num_scales = [0,1,2,3,4,5]
        self.input_ctrs = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            MLP(n_map, n_map,  bias=False, act=None, norm=norm, group_num=group_num, res_type=False)
        )
        self.input_feats = nn.Sequential(
            nn.Linear(2, n_map)  if not add_theta_info else nn.Linear(3, n_map),
            nn.ReLU(inplace=True),
            MLP(n_map, n_map,  bias=False, act=None, norm=norm, group_num=group_num, res_type=False)
        )

        keys = ["ctr", "norm", "ctr2", "left", "right"] + ["pre0", "pre1", "pre2", "pre3", "pre4", "pre5"] + ["suc0", "suc1", "suc2", "suc3", "suc4", "suc5"]

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for _ in range(self.num_blocks):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(group_num, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(MLP(n_map, n_map,  bias=False, act=None, norm=norm, group_num=group_num, res_type=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

        self.apply(self._init_weights)

    def forward(self, graph_ctrs, graph_feats, graph_pre, graph_suc, graph_left, graph_right):
        if (
            len(graph_feats) == 0
            or len(graph_pre[-1]["u"]) == 0
            or len(graph_suc[-1]["u"]) == 0
        ):
            print(len(graph_feats))
            print(len(graph_pre[-1]["u"]))
            print(len(graph_suc["u"]))
            temp = graph_feats
            return (
                temp.new().resize_(0),
                [temp.new().long().resize_(0)],
                temp.new().resize_(0),
            )


        ctrs = torch.cat(graph_ctrs, 0)
        feat = self.input_ctrs(ctrs)
        feat += self.input_feats(graph_feats)
        feat = self.relu(feat)

        identity = feat

        for i in range(self.num_blocks):
            temp = self.fuse["ctr"][i](feat)
            for idx in self.num_scales:

                temp.index_add_(
                    0,
                    graph_pre[idx]["u"],
                    self.fuse["pre"+str(idx)][i](feat[graph_pre[idx]["v"].long()]),
                )

                temp.index_add_(
                    0,
                    graph_suc[idx]["u"],
                    self.fuse["suc"+str(idx)][i](feat[graph_suc[idx]["v"].long()]),
                )

            if len(graph_left["u"] > 0):
                temp.index_add_(
                    0,
                    graph_left["u"],
                    self.fuse["left"][i](feat[graph_left["v"].long()]),
                )
            if len(graph_right["u"] > 0):
                temp.index_add_(
                    0,
                    graph_right["u"],
                    self.fuse["right"][i](feat[graph_right["v"].long()]),
                )
            
            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += identity
            feat = self.relu(feat)
            identity = feat

        return feat

    def _init_weights(self, m):
        common_init(m)


class FeatureHardAdd(nn.Module):
    def __init__(self, agt_feature=128, ctx_feature=128):
        super(FeatureHardAdd, self).__init__()
        norm = "GN"
        group_num = 1

        self.dist = nn.Sequential(
            nn.Linear(2, ctx_feature),
            nn.ReLU(inplace=True),
            MLP(ctx_feature, ctx_feature,  bias=False, act='relu', norm=norm, group_num=group_num, res_type=False)
        )
        self.query = MLP(agt_feature, ctx_feature,  bias=False, act='relu', norm=norm, group_num=group_num, res_type=False)

        self.ctx = nn.Sequential(
            MLP(3*ctx_feature, agt_feature,  bias=False, act='relu', norm=norm, group_num=group_num, res_type=False),
            nn.Linear(agt_feature, agt_feature, bias=False)
        )

        self.agt = nn.Linear(agt_feature, agt_feature, bias=False)
        self.norm = nn.GroupNorm(gcd(group_num, agt_feature), agt_feature)
        self.linear = MLP(agt_feature, agt_feature,  bias=False, act=None, norm=norm, group_num=group_num, res_type=False)
        self.relu = nn.ReLU(inplace=True)

        self.apply(self._init_weights)

    def forward(self, agts, ctx, distance, hi, wi):
        identity = agts
        dist = self.dist(distance)
        query = self.query(agts[hi])

        ctx = ctx[wi]

        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)

        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += identity
        agts = self.relu(agts)
        return agts

    def _init_weights(self, m):
        common_init(m)

class A2M(nn.Module):

    def __init__(self, n_actor=128, n_map=128):
        super(A2M, self).__init__()
        norm = "GN"
        group_num = 1 
        self.num_att_block = 2      

        self.meta = MLP(n_map+4, n_map,  bias=False, act='relu', norm=norm, group_num=group_num, res_type=False)

        att = []
        for _ in range(self.num_att_block):
            att.append(FeatureHardAdd(n_map, n_actor))
        self.att = nn.ModuleList(att)

        self.apply(self._init_weights)

    def forward(self, feat,  graph_turn, graph_control, graph_intersect, actors, distance, hi, wi):
        meta = torch.cat((graph_turn,
                                            graph_control.unsqueeze(1),
                                            graph_intersect.unsqueeze(1),
                                            ),
                                            1
            )

        feat = self.meta(torch.cat((feat, meta), 1))
        
        for i in range(self.num_att_block):
            feat = self.att[i](feat, actors, distance, hi, wi)

        return feat

    def _init_weights(self, m):
        common_init(m)


class M2M(nn.Module):

    def __init__(self, n_map=128):
        super(M2M, self).__init__()
        norm = "GN"
        group_num = 1
        self.num_blocks = 4
        self.num_scales = [0,1,2,3,4,5]

        keys = ["ctr", "norm", "ctr2", "left", "right", "pre0", "suc0", "pre1", "suc1", "pre2", "suc2", "pre3", "suc3", "pre4", "suc4", "pre5", "suc5"]

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for _ in range(self.num_blocks):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(group_num, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(MLP(n_map, n_map,  bias=False, act=None, norm=norm, group_num=group_num, res_type=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

        self.apply(self._init_weights)

    def forward(self, feat, graph_pre, graph_suc, graph_left, graph_right):
        identity = feat
        for i in range(self.num_blocks):
            temp = self.fuse["ctr"][i](feat)
            for idx in self.num_scales:
                temp.index_add_(
                    0,
                    graph_pre[idx]["u"],
                    self.fuse["pre"+str(idx)][i](feat[graph_pre[idx]["v"]]),
                )

                temp.index_add_(
                    0,
                    graph_suc[idx]["u"],
                    self.fuse["suc"+str(idx)][i](feat[graph_suc[idx]["v"]]),
                )

            if len(graph_left["u"] > 0):
                temp.index_add_(
                    0,
                    graph_left["u"],
                    self.fuse["left"][i](feat[graph_left["v"]]),
                )
            if len(graph_right["u"] > 0):
                temp.index_add_(
                    0,
                    graph_right["u"],
                    self.fuse["right"][i](feat[graph_right["v"]]),
                )
            
            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += identity
            feat = self.relu(feat)
            identity = feat

        return feat

    def _init_weights(self, m):
        common_init(m)

class M2A(nn.Module):

    def __init__(self, n_map=128, n_actor=128):
        super(M2A, self).__init__()
        self.num_att_block = 2      

        att = []
        for _ in range(self.num_att_block):
            att.append(FeatureHardAdd(n_actor, n_map))
        self.att = nn.ModuleList(att)

        self.apply(self._init_weights)

    def forward(self, actors, nodes, distance, hi, wi):

        for i in range(self.num_att_block):
            actors = self.att[i](actors, nodes, distance, hi, wi)
        return actors

    def _init_weights(self, m):
        common_init(m)


class A2A(nn.Module):

    def __init__(self, n_actor=128):
        super(A2A, self).__init__()
        self.num_att_block = 2

        att = []
        for _ in range(self.num_att_block):
            att.append(FeatureHardAdd(n_actor, n_actor))
        self.att = nn.ModuleList(att)

        self.apply(self._init_weights)

    def forward(self, actors, distance, hi, wi):
        for i in range(self.num_att_block):
            actors = self.att[i](actors, actors, distance, hi, wi)
        return actors

    def _init_weights(self, m):
        common_init(m)


class AttDest(nn.Module):

    def __init__(self, n_actor=128, num_mods=6):
        super(AttDest, self).__init__()
        norm = "GN"
        act = "relu"
        group_num = 1
        self.n_actor = n_actor
        self.num_mods = num_mods

        self.dist = nn.Sequential(
            nn.Linear(2, n_actor),
            nn.ReLU(inplace=True),
            MLP(n_actor, n_actor,  bias=False, act=act, norm=norm, group_num=group_num, res_type=False)
        )

        self.agt = MLP(2*n_actor, n_actor,  bias=False, act=act, norm=norm, group_num=group_num, res_type=False)
         
        self.apply(self._init_weights)

    def forward(self, agts, agt_ctrs,  dest_ctrs):

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, self.num_mods, 1).view(-1, self.n_actor)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts

    def _init_weights(self, m):
        common_init(m)

class FilterDistance(nn.Module):

    def __init__(self, dist_th=7):
        super(FilterDistance, self).__init__()
        self.dist_th = dist_th

    def forward(self, agt_idcs, agt_ctrs, ctx_idcs, ctx_ctrs):
        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= self.dist_th

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue
            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count) 
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)


        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        return dist, hi, wi
