'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''


from torch.utils.data import Dataset
import os
import pickle
import zlib

from data_utils import Collater_Lanegcn

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
