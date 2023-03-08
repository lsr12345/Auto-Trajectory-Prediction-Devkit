'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''
import torch

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from utils.standard_tools import recursiveToTensor

class Collater_Lanegcn():

    def __init__(self, mod=6):
        self.mod = mod

        self.input_keys = ["actors", "graph"]
        self.concat_batch = ["feats", "graph_feats", "graph_turn", "graph_control", "graph_intersect"]
        self.nonconcat_bath = ["actor_ctrs", "graph_ctrs"]

        self.common_keys = ["rot", "orig", "theta"]

        self.output_keys = "gt"
        self.output_sub_keys = ["gt_preds", "has_preds"]

    def __call__(self, batch_data):
        batch_data = recursiveToTensor(batch_data)
        batch_size = len(batch_data)
        num_actors = [(x["actors"]["feats"].shape[0]) for x in batch_data]

        actor_idcs = []
        actor_count = 0

        node_idcs = []
        node_count = 0
        node_counts = []

        for i in range(batch_size):
            a_idcs = torch.arange(actor_count, actor_count + num_actors[i])

            actor_idcs.append(a_idcs)
            actor_count += num_actors[i]

            node_counts.append(node_count)

            n_idcs = torch.arange(node_count, node_count + batch_data[i]["graph"]["num_nodes"])

            node_idcs.append(n_idcs)
            node_count = node_count + batch_data[i]["graph"]["num_nodes"]

        return_input_batch = dict()

        for key in self.common_keys:
            return_input_batch[key] = []
        for key in self.common_keys:
            return_input_batch[key] = [x[key].float() for x in batch_data]


        for key in self.input_keys:
            return_input_batch[key] = dict()
            for ke in batch_data[0][key]:
                if ke != "num_nodes":
                    return_input_batch[key][ke] = dict()

                if ke in self.concat_batch:
                    return_input_batch[key][ke] = torch.cat([x[key][ke] for x in batch_data], 0)

                elif ke in self.nonconcat_bath:

                    return_input_batch[key][ke] = [x[key][ke] for x in batch_data]
                
                elif ke in ["graph_pre", "graph_suc"]:
                    return_input_batch[key][ke] = []
                    for ii in range(self.mod):
                        return_input_batch[key][ke].append(dict())
                        for k in ['v', 'u']:
 
                            return_input_batch[key][ke][ii][k] = torch.cat([batch_data[iii][key][ke][ii][k].to(torch.int32) + node_counts[iii]for iii in range(batch_size)], 0).long()              
                elif ke in ["graph_left", "graph_right"]:
                    for k in ['v', 'u']:
                        temp = [batch_data[i][key][ke][k].to(torch.int32) + node_counts[i] for i in range(batch_size)]
                        temp = [
                            x if x.dim() > 0 else batch_data[0]["graph"]["graph_pre"]["u"].new().resize_(0)  
                            for x in temp
                        ]
                        
                        return_input_batch[key][ke][k] = torch.cat(temp).long()
                    
                elif ke in ["num_nodes", "lane_idcs"]:
                    continue

                else:
                    raise NotImplementedError('check data key:  {} '.format(ke))

        return_input_batch["actors"]["actor_idcs"] = actor_idcs
        return_input_batch["graph"]["graph_idcs"] = node_idcs

        return_output_batch = dict()

        for ke in self.output_sub_keys:
            return_output_batch[ke] = [x[self.output_keys][ke] for x in batch_data]


        for key in self.common_keys:
            return_output_batch[key] = [x[key].float() for x in batch_data]
        
        return return_input_batch, return_output_batch


























