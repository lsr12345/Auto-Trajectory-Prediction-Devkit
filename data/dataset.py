'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''

from torch.utils.data import Dataset
import numpy as np
import math
import os
import pickle
import zlib

from argoverse.map_representation.map_api import ArgoverseMap
from data_utils import Collater_Lanegcn, Collater_VectorNet

class Argoverse_forecast_LanegcnDataset(Dataset):
    
    def __init__(
            self, 
            data_dir="/xx.pkl",
            dtype="train"
    ):
        super().__init__()

        keys = ["actors",  "graph", "gt", "rot", "orig", "theta"]
        self.file_path = []
        file_list = os.listdir(data_dir)
        
        for f in file_list:
            self.file_path.append(os.path.join(data_dir, f))
        self.keys = keys
    
    def __getitem__(self, i):
        data = self.readfile( self.file_path[i])
        n_store = dict(data)

        new_data = dict()
        for key in self.keys:
            new_data[key] = n_store[key]
        return new_data
        
    def __len__(self):
        return len(self.file_path)
    
    def readfile(self, path):
        with open(path, 'rb') as fr:
            data = zlib.decompress(pickle.load(fr))
            data = pickle.loads(data)
        return data

    def get_collate_fn(self):
        return Collater_Lanegcn(mod=6)

class Argoverse_forecast_VectorNetDataset(Dataset):
    
    def __init__(
            self, 
            data_dir="./Argoverse/forecast/train/",
            dtype="train",
            keys=["matrix_objects","matrix_lanes", "labels",  "labels_is_valid", "agent_trainable"],
            hidden_size=16,
            future_frame_num=30,
            save_pkl=False,
            save_out_dir="./prediction/pkl"
    ):
        super().__init__()
        self.save_pkl = save_pkl
        self.save_out_dir = save_out_dir

        self.file_path = []
        file_list = os.listdir(data_dir)
        for f in file_list:
            if f.endswith(".csv"):
                self.file_path.append(os.path.join(data_dir, f))
        self.hidden_size = hidden_size
        self.future_frame_num = future_frame_num
        self.am = ArgoverseMap()

        self.keys = keys
        self.TIMESTAMP = 0
        self.TRACK_ID = 1
        self.OBJECT_TYPE = 2
        self.X = 3
        self.Y = 4
        self.CITY_NAME = 5
        self.VECTOR_PRE_X = 0
        self.VECTOR_PRE_Y = 1
        self.VECTOR_X = 2
        self.VECTOR_Y = 3
    
    def __getitem__(self, i):
        lines = self.readfile(self.file_path[i])
        mapping = self.argoverse_get_instance(self.am, lines, self.file_path[i], self.hidden_size, self.future_frame_num)

        if self.save_pkl:
            if not os.path.exists(self.save_out_dir):
                os.mkdir(self.save_out_dir)
            save_path = os.path.join(self.save_out_dir, "{}.pkl".format(os.path.basename(self.file_path[i])))
            with open(save_path, 'wb') as f:
                mapping_zlib = zlib.compress(pickle.dumps(mapping))
                pickle.dump(mapping_zlib, f)

        new_data = dict()
        for key in self.keys:
            new_data[key] = mapping[key]
        return new_data
        
    def __len__(self):
        return len(self.file_path)
    
    def readfile(self, path):
        with open(path, "r", encoding='utf-8') as fr:
            lines = fr.readlines()[1:]
        return lines

    def get_collate_fn(self):
        return Collater_VectorNet()

    def argoverse_get_instance(self, am, lines, file_name, hidden_size=32, future_frame_num=30):
        vector_num = 0
        id2info = {}
        mapping = {}
        mapping['file_name'] = file_name
        for i, line in enumerate(lines):
            line = line.strip().split(',')
            if i == 0:
                mapping['start_time'] = float(line[self.TIMESTAMP])
                mapping['city_name'] = line[self.CITY_NAME]

            line[self.TIMESTAMP] = float(line[self.TIMESTAMP]) - mapping['start_time']
            line[self.X] = float(line[self.X])
            line[self.Y] = float(line[self.Y])
            id = line[self.TRACK_ID]

            if line[self.OBJECT_TYPE] == 'AV' or line[self.OBJECT_TYPE] == 'AGENT':
                line[self.TRACK_ID] = line[self.OBJECT_TYPE]

            if line[self.TRACK_ID] in id2info:
                id2info[line[self.TRACK_ID]].append(line)
                vector_num += 1
            else:
                id2info[line[self.TRACK_ID]] = [line]

            if line[self.OBJECT_TYPE] == 'AGENT' and len(id2info['AGENT']) == 20:
                assert 'AV' in id2info
                assert 'cent_x' not in mapping
                agent_lines = id2info['AGENT']          
                mapping['cent_x'] = agent_lines[-1][self.X]                      
                mapping['cent_y'] = agent_lines[-1][self.Y]
                mapping['agent_pred_index'] = len(agent_lines)      
                mapping['two_seconds'] = line[self.TIMESTAMP]
                
                span = agent_lines[-6:]
                angles = []
                interval= 2     
                for j in range(len(span)):
                    if j + interval < len(span):
                        der_x, der_y = span[j + interval][self.X] - span[j][self.X], span[j + interval][self.Y] - span[j][self.Y]
                        angles.append([der_x, der_y])
        angles = np.array(angles)
        der_x, der_y = np.mean(angles, axis=0)
        angle = -self.get_angle(der_x, der_y) + math.radians(90)
        
        mapping['angle'] = angle

        for i in id2info:
            info = id2info[i]
            for line in info:
                line[self.X], line[self.Y] = self.rotate(line[self.X] - mapping['cent_x'], line[self.Y] - mapping['cent_y'], angle)
        
        keys = list(id2info.keys())
        assert 'AV' in keys
        assert 'AGENT' in keys
        keys.remove('AV')
        keys.remove('AGENT')
        keys = ['AV', 'AGENT'] + keys
        
        two_seconds = mapping['two_seconds']
        mapping['trajs'] = []
        mapping['agents'] = []

        lane_ids = am.get_lane_ids_in_xy_bbox(mapping['cent_x'], mapping['cent_y'], mapping['city_name'], query_search_range_manhattan=50)
        centerlines = [am.get_lane_segment_centerline(lane_id, mapping['city_name']) for lane_id in lane_ids]
        
        centerlines = [centerline[:, :2].copy() for centerline in centerlines]
        for index_centerline, centerline in enumerate(centerlines):
            for i, point in enumerate(centerline):
                point[0], point[1] = self.rotate(point[0] - mapping['cent_x'], point[1] -  mapping['cent_y'], mapping['angle'])

        matrix_objects_vectors = []
        agent_trajs = []
        agent_trainable_mask = []
        agent_future_trajs = []
        agent_future_mask = []

        for index, key in enumerate(keys):
            info = id2info[key]
            info_valid = False
            vectors = []
            trajs = []
            future = np.zeros(shape=(30, 2))
            future_mask = [0] * 30
            future_index = 0
            
            for _, line in enumerate(info):
                if line[self.TIMESTAMP] > two_seconds+1e-5:
                    if not info_valid:
                        break
                    else:
                        future[future_index, 0] = line[self.X]
                        future[future_index, 1] = line[self.Y]
                        future_mask[future_index] = 1
                        future_index += 1
                else:
                    trajs.append([line[self.X], line[self.Y]])
                    if len(trajs) > 1:
                        info_valid = True
            if info_valid:
                agent_trainable_mask.append(0)
            if len(trajs) == 20 and info_valid:
                agent_trajs.append(trajs)
                agent_trainable_mask[-1] = 1
                agent_future_trajs.append(future)
                agent_future_mask.append(future_mask)
            
            s_x, s_y = 0,0
            e_x, e_y = 0,0
            for i, line in enumerate(info):
                if i == 0:
                    s_x, s_y =  line[self.X], line[self.Y]
                    line_pre = line
                if line[self.TIMESTAMP] > two_seconds+1e-5:
                    e_x, e_y =  line_pre[self.X], line_pre[self.Y]
                    break
                line_pre = line

            for i, line in enumerate(info):
                if line[self.TIMESTAMP] > two_seconds+1e-5:
                    break
                x, y = line[self.X], line[self.Y]
                if i > 0:
                    vector = [s_x, s_y, e_x, e_y, x-line_pre[self.X], y-line_pre[self.Y], line[self.TIMESTAMP], line[self.OBJECT_TYPE] == 'AV',
                              line[self.OBJECT_TYPE] == 'AGENT', line[self.OBJECT_TYPE] == 'OTHERS',  mapping['angle'], index, i-1]
                    vectors.append(self.get_pad_vector(vector, hidden_size=hidden_size))      
                line_pre = line

            if len(vectors) > 0:
                matrix_objects_vectors.append(vectors)
        
        assert agent_trainable_mask[0] == 1
        assert agent_trainable_mask[1] == 1

        matrix_objects = np.zeros((len(matrix_objects_vectors), 20, self.hidden_size))
        for object_index, vectors in enumerate(matrix_objects_vectors):
            matrix_objects[object_index, :len(vectors)] = np.array(vectors)

        matrix_lanes_vectors = []
        
        for index_centerline, centerline in enumerate(centerlines):
            lanes_vectors = []
            assert 2 <= len(centerline) <= 10, "{}".format(len(centerline))
            lane_id = lane_ids[index_centerline]
            lane_segment = am.city_lane_centerlines_dict[mapping['city_name']][lane_id]
            assert len(centerline) >= 2

            for i, point in enumerate(centerline):
                if i > 0:
                    vector = [0] * hidden_size
                    vector[0], vector[1] = point_pre[0], point_pre[1]        
                    vector[2], vector[3] = point[0], point[1]    
                    vector[4] = 1
                    vector[5] = i-1
                    vector[6] = index_centerline
                    vector[7] = 1 if lane_segment.has_traffic_control else -1
                    vector[8] = 1 if lane_segment.turn_direction == 'RIGHT' else 0
                    vector[9] = -1 if lane_segment.turn_direction == 'LEFT' else 0
                    vector[10] = 1 if lane_segment.is_intersection else -1
                    if i >= 2:
                        point_pre_pre = centerline[i - 2]
                    else:
                        point_pre_pre = (2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                    vector[11] = point_pre_pre[0]
                    vector[12] = point_pre_pre[1]
                    vector[13] = mapping['angle']

                    lanes_vectors.append(vector)
                point_pre = point

            if len(lanes_vectors) > 0:
                matrix_lanes_vectors.append(lanes_vectors)

        matrix_lanes = np.zeros((len(matrix_lanes_vectors),10, self.hidden_size))
        for lane_index, lanes_vectors in enumerate(matrix_lanes_vectors):
            matrix_lanes[lane_index, :len(lanes_vectors)] = np.array(lanes_vectors)
    
        mapping.update(dict(
            matrix_objects=matrix_objects,
            matrix_lanes=matrix_lanes,
            agent_trajs = agent_trajs,
            agent_trainable = np.array(agent_trainable_mask),       
            labels = np.array(agent_future_trajs),      
            labels_is_valid=np.array(agent_future_mask, dtype=np.int64),     
            eval_time=30,
        ))
        return mapping

    def get_angle(self, x, y):
        return math.atan2(y, x)

    def rotate(self, x, y, angle):
        res_x = x * math.cos(angle) - y * math.sin(angle)
        res_y = x * math.sin(angle) + y * math.cos(angle)
        return res_x, res_y

    def get_pad_vector(self, li, hidden_size=128):
        assert len(li) <= hidden_size      
        li.extend([0] * (hidden_size - len(li)))
        return li
