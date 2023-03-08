'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''
# coding: utf-8
import torch
from torch.utils.data import DataLoader

from loguru import logger
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from data.dataset import Argoverse_forecast_LanegcnDataset

class Data_loader():
    def __init__(self, config, args): 
        self.is_main_process = True if  args.rank== 0 else False
        self.config = config
        self.train_list = config.get('train_list', None)
        self.test_list = config.get('test_list', None)
        self.dataset_name = self.config['dataset_name']
        
    def get_train(self, distributed=False, nprocs=1):
        if self.is_main_process:
            logger.info("Trian Dataset name: {}".format(self.dataset_name))
        if self.dataset_name == 'Argoverse_forecast_Lanegcn':
            train_dataset = Argoverse_forecast_LanegcnDataset(self.config['train_data_dir'], dtype="train")
        else:
            raise NotImplementedError('{} dataset_name not supported.'.format(self.dataset_name))

        if self.is_main_process:
            logger.info("Train Dataset samples: {}".format(len(train_dataset)))

        if not distributed:
            return DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=self.config.get('shuffle', True),
                            num_workers=self.config['num_workers'], collate_fn=train_dataset.get_collate_fn())

        else:
            assert self.config['batch_size']  % nprocs == 0
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            sampler_flag = self.config.get('sampler', True)
            train_loader = DataLoader(train_dataset,
                                    batch_size=self.config['batch_size']  // nprocs,
                                    num_workers=max(self.config['num_workers'] // nprocs, 1),
                                    pin_memory=True,
                                    # shuffle=True,
                                    sampler=train_sampler if sampler_flag else None,
                                    collate_fn=train_dataset.get_collate_fn())
            return train_loader, train_sampler

    
    def get_test(self, distributed=False, nprocs=1):
        if self.is_main_process:
            logger.info("Test Dataset name: {}".format(self.dataset_name))


        if self.dataset_name == 'Argoverse_forecast_Lanegcn':
            test_dataset = Argoverse_forecast_LanegcnDataset(self.config['test_data_dir'], dtype="val")
        else:
            raise NotImplementedError('{} dataset_name not supported.'.format(self.dataset_name))
        if self.is_main_process:
            logger.info("Test Dataset samples: {}".format(len(test_dataset)))
        
        if not distributed:
            return DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False,
                            num_workers=self.config['num_workers'], collate_fn=test_dataset.get_collate_fn())
        else:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            sampler_flag = self.config.get('sampler', True)
            test_loader = DataLoader(test_dataset,
                                    batch_size=self.config['batch_size']  // nprocs,
                                    num_workers=max(self.config['num_workers'] // nprocs, 1),
                                    pin_memory=True,
                                    # shuffle=False,
                                    sampler=test_sampler if sampler_flag else None,
                                    collate_fn=test_dataset.get_collate_fn()
                                    )
            return test_loader, test_sampler


