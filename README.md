# MUFold-ss-pytorch
## Introduction
MUFold-ss (https://arxiv.org/pdf/1709.06165.pdf) protein secondary structure prediction implementation in PyTorch. Note that
the Conv11 in the final Struct2Struct network is not implemented.

Run `python mode.py exp1` to train the network and validate after each epoch. My data is not shareable but you can follow the
links provided in the paper to download the Cullpdb data. Modify `data_loading.py` to properly load that version data.

## Dependency
1. Python 2.7
2. Pytorch 0.2.0
3. tensorboard-pytorch

## To do 
* Improve the argparse and file management in model.py.
* Properly set up saving and restoring of the model.
* Try out different hyperparameters and optimizers.

## Misc
Several bugs in PyTorch. Conv1d only accepts DoubleTensor and BatchNorm1d only accepts float.
