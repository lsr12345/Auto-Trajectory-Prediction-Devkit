'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example: argoverse v1 datasets preprocess

'''

import numpy as np
import argparse
import math
import copy
from scipy import sparse
from tqdm import tqdm
import pickle
import zlib

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def arg_parser():
    parser = argparse.ArgumentParser("preprocess datasets")
    
    parser.add_argument(
        "-i", "--input_file", required=True, type=str, help="what's the input file"
    )
    parser.add_argument(
        "-t", "--task", default="argoverse_forecast", type=str, help="what's the task"
    )
    parser.add_argument(
        "-m", "--mode", default="val", type=str, help="train val test tiny"
    )
    parser.add_argument(
        "-b", "--batch", default=32, type=int, help="train val test tiny"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default='../lanegcn_preprocess_dataset',
        type=str,
        help="save dir",
    )

    return parser

def from_numpy(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data

def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

def to_numpy(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data

def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data

def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

def gpu(data):
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous()
    return data

def preprocess(graph, cross_dist):
    left, right = dict(), dict()
    
    lane_idcs = graph['lane_idcs']
    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))

    hi = torch.arange(num_nodes).long().view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long()

    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.numpy().astype(np.int16)
        right['v'] = vi.numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    out = dict()
    out['left'] = left
    out['right'] = right
    out['idx'] = graph['idx']
    return out

def fromloader_save(data_loader, batch_size, output_dir):
    for i, data in enumerate(tqdm(data_loader)):
        data = [x for x in data]
        for ii, n_store in enumerate(data):
            save_path = output_dir + "{}.pkl".format(i*batch_size+ii)
            with open(save_path, 'wb') as f:
                n_store_zlib = zlib.compress(pickle.dumps(n_store))
                pickle.dump(n_store_zlib, f)

class ArgoDataset(Dataset):
    def __init__(self, split, mean_theta=False):
        self.avl = ArgoverseForecastingLoader(split)
        self.avl.seq_list = sorted(self.avl.seq_list)
        self.am = ArgoverseMap()
        print('Total number of sequences:',len(self.avl))
        self.keys = [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph"]

        self.graph_keys = ['lane_idcs',
                        'ctrs',
                        'pre_pairs',
                        'suc_pairs',
                        'left_pairs',
                        'right_pairs',
                        'feats']


        self.mean_theta = mean_theta

    def __getitem__(self, idx):
        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data, mean_theta=self.mean_theta)
        data['idx'] = idx

        data['graph'] = self.get_lane_graph(data)
        store = dict()
        for key in self.keys:
            store[key] = to_numpy(data[key])
            if key in ["graph"]:
                store[key] = to_int16(store[key])
        graph = dict()
        for key in self.graph_keys:
            graph[key] = ref_copy(store['graph'][key])
        graph['idx'] = idx

        graph = preprocess(to_long(from_numpy(graph)), 6)
        store['graph']['left'] = graph['left']
        store['graph']['right'] = graph['right']

        n_store = dict()
        n_store["actors"] = dict()
        n_store["graph"] = dict()
        n_store["gt"] = dict()

        n_store["actors"]["feats"] = store["feats"]
        n_store["actors"]["actor_ctrs"] = store["ctrs"]

        n_store["graph"]["graph_ctrs"] = store["graph"]["ctrs"]
        n_store["graph"]["graph_feats"] = store["graph"]["feats"]
        n_store["graph"]["num_nodes"] = store["graph"]["num_nodes"]
        n_store["graph"]["graph_turn"] = store["graph"]["turn"]
        n_store["graph"]["graph_control"] = store["graph"]["control"]
        n_store["graph"]["graph_intersect"] = store["graph"]["intersect"]
        n_store["graph"]["graph_pre"] = store["graph"]["pre"]
        n_store["graph"]["graph_suc"] = store["graph"]["suc"]
        n_store["graph"]["graph_left"] = store["graph"]["left"]
        n_store["graph"]["graph_right"] = store["graph"]["right"]
        n_store["graph"]["lane_idcs"] = store["graph"]["lane_idcs"]

        n_store["gt"]["gt_preds"] = store["gt_preds"]
        n_store["gt"]["has_preds"] = store["has_preds"]

        n_store["rot"] = store["rot"]
        n_store["orig"] = store["orig"]
        n_store["theta"] = np.array(store["theta"])

        return n_store

    def __len__(self):
        return len(self.avl)

    def read_argo_data(self, idx):
        city = self.avl[idx].city
        df = self.avl[idx].seq_df
        
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)
        
        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)
        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]
        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]
        
        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        return data
    
    def get_obj_feats(self, data, mean_theta=False):
        orig = data['trajs'][0][19].copy().astype(np.float32)

        if not mean_theta:
            pre = data['trajs'][0][18] - orig
            theta = np.pi - np.arctan2(pre[1], pre[0])
        else:
            pre = data['trajs'][0][14:19] - data['trajs'][0][18]

            pre_thetas = [(math.pi - math.atan2(p[1], p[0])) for p in pre]
            pre_thetas = [int(i * 180/math.pi) for i in pre_thetas]
            theta = self.calc_angles_mean(pre_thetas)
            theta = theta / 180 * math.pi


        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, ctrs, gt_preds, has_preds = [], [], [], []
        ctrs_3 = []
        for traj, step in zip(data['trajs'], data['steps']):
            if 19 not in step:
                continue

            gt_pred = np.zeros((30, 2), np.float32)
            has_pred = np.zeros(30, bool)
            future_mask = np.logical_and(step >= 20, step < 50)
            post_step = step[future_mask] - 20
            post_traj = traj[future_mask]
    
            gt_pred[post_step] = post_traj
            has_pred[post_step] = 1
            
            obs_mask = step < 20
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]
            
            for i in range(len(step)):
                if step[i] == 20 - len(step) + i:
                    break
            step = step[i:]
            traj = traj[i:]

            feat = np.zeros((20, 3), np.float32)
            feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat[step, 2] = 1.0

            x_min, x_max, y_min, y_max = [-100.0, 100.0, -100.0, 100.0]
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue
            
            ctrs.append(feat[-1, :2].copy())

            ctrs_3.append(feat[-3:, :2].copy())

            feat[1:, :2] -= feat[:-1, :2]
            feat[step[0], :2] = 0
            feats.append(feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, bool)


        ctrs_3 = np.asarray(ctrs_3, np.float32)

        data['feats'] = feats
        data['ctrs'] = ctrs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds

        data['ctrs_3'] = ctrs_3

        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = [-100.0, 100.0, -100.0, 100.0]
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)
        
        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane
            
        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1
            
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))
            
            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))
            
        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count
        
        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]
            idcs = node_idcs[i]
            
            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if lane.predecessors is not None:
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])
                    
            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)


        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        
        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)
        
        for key in ['pre', 'suc']:
            graph[key] += self.dilated_nbrs(graph[key][0], graph['num_nodes'], 6) 
        return graph

    def dilated_nbrs(self, nbr, num_nodes, num_scales):
        data = np.ones(len(nbr['u']), bool)
        csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

        mat = csr
        nbrs = []
        for i in range(1, num_scales):
            mat = mat * mat

            nbr = dict()
            coo = mat.tocoo()
            nbr['u'] = coo.row.astype(np.int64)
            nbr['v'] = coo.col.astype(np.int64)
            nbrs.append(nbr)
        return nbrs

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


def main():
    args = arg_parser().parse_args()
    input_dir = args.input_file
    output_dir = args.output_dir
    mode = args.mode
    output_dir = os.path.join(output_dir, mode)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task = args.task
    assert task in ["argoverse_forecast"]
    assert mode in ["train", "val", "test", "tiny"]
    num_dict = {"train": 205942, "val":39472, "test": 78143, "tiny":64}
    batch_size = args.batch
    print(num_dict)
    print('mode: {}, dataset samples: '.format(mode ))
    dataset = ArgoDataset(input_dir, mean_theta=False)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=batch_size//8,
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False,
    )    


    save_path = os.path.join(output_dir, "{}_{}_".format(task, mode))
    fromloader_save(train_loader, batch_size, save_path)


if __name__ == '__main__':
    main()