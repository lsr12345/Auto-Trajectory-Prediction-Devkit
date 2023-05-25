
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
import math

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.utils.ops import MLP
from tools.nninit import common_init

class PolyGraph(nn.Module):
    def __init__(self, hidden_size=64, depth=3,  norm = "LN", act = "relu"):
        super(PolyGraph, self).__init__()
        self.mlp_0 = MLP(num_in=16, num_out=hidden_size, norm=norm, act=act, bias=True)
        self.mlp_1 = MLP(hidden_size, norm=norm, act=act, bias=True)
        self.sa_layers = nn.ModuleList([SA_Layer(hidden_size, num_attention_heads=2) for _ in range(depth)])
        self.LayerNorms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(depth)])

        self.apply(self._init_weights)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.mlp_0(hidden_states)
        hidden_states = self.mlp_1(hidden_states)
        for i, layer in enumerate(self.sa_layers):           
            temp = hidden_states
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.LayerNorms[i](hidden_states)
        
        return torch.max(hidden_states, dim=1)[0]

    def _init_weights(self, m):
        common_init(m)

class SA_Layer(nn.Module):
    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1):
        super(SA_Layer, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.apply(self._init_weights)

    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        
        x = x.view(*sz)
        return x.permute(0, 2, 1, 3)

    def forward(self,hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

    def _init_weights(self, m):
        common_init(m)

class SACat_Layer(nn.Module):

    def __init__(self, hidden_size):
        super(SACat_Layer, self).__init__()
        self.global_graph = SA_Layer(hidden_size, hidden_size // 2)
        self.global_graph2 = SA_Layer(hidden_size, hidden_size // 2)

        self.apply(self._init_weights)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = torch.cat([self.global_graph(hidden_states, attention_mask),
                                   self.global_graph2(hidden_states, attention_mask)], dim=-1)
        return hidden_states

    def _init_weights(self, m):
        common_init(m)
