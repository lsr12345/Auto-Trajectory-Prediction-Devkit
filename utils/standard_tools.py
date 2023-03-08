import numpy as np
import torch


def recursiveToTensor(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = recursiveToTensor(data[key])
    elif isinstance(data, list) or isinstance(data, tuple):
        data = [recursiveToTensor(x) for x in data]
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif  torch.is_tensor(data):
        return data
    return data

def togpu(data):
    if isinstance(data, list) or isinstance(data, tuple):
        data = [togpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:togpu(_data) for key,_data in data.items()}

    else:
        if not torch.is_tensor(data):
            data = torch.tensor(data)
        data = data.contiguous().cuda(non_blocking=True)
    return data

def tolong(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = tolong(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [tolong(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

def recursiveToNumpy(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = recursiveToTensor(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = np.array(data)
    return data