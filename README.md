# Auto-Trajectory-Prediction-Devkit

This repository contains related algorithms in prediction fields: 
- [x] LaneGCN
- [x] LaneGCN checkpoints
- [ ] Vectornet
- [ ] Vectornet checkpoints
- [ ] TNT
- [ ] TNT checkpoints
- [ ] mmtransformer
- [ ] mmtransformer checkpoints


## Table of Contents
- [Auto-Trajectory-Prediction-Devkit](#auto-trajectory-prediction-devkit)
  - [Table of Contents](#table-of-contents)
  - [Methods Difference](#methods-difference)
  - [How to Use](#how-to-use)
  - [Requirements](#requirements)

## Methods Difference
| Method | ADE_1 | FDE_1| MR_1| mADE | mFDE | mMR |
| :---: | :---: | :---: | :---:|:---:| :---: | :---: |
| LaneGCN | 1.31 | 2.88 | - | 0.70 | 1.0 | - |
- LaneGCN
  - More lightweight data preprocessing(official need more 40G+ memory)
  - Optimized part of GCN attention module
  - Add a module to supervise predict angles
  - Easy to distribute training

## How to Use

```bash
# for lanegcn:
$ cd ./data
$ python lanegcn_preprocess.py -i /Argoverse/train -m train
$ python lanegcn_preprocess.py -i /Argoverse/val -m val
$ python train_ddp.py -f ./config/Config_Lanegcn.yaml
```

For details about how to configure related algorithms, see examples.


## Requirements

* `requirements.txt`