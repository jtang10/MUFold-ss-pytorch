from __future__ import print_function, division

import time
import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from data_loading import Protein_Dataset
from model import MUFold_ss

SetOf7604Proteins_path = '../data/SetOf7604Proteins/'
CASP11_path = '../data/CASP11/'
CASP12_path = '../data/CASP12/'
trainList_addr = 'trainList'
validList_addr = 'validList'
testList_addr = 'testList'
proteinList_addr = 'proteinList'

use_cuda = torch.cuda.is_available()

test_dataset = Protein_Dataset(SetOf7604Proteins_path, testList_addr, padding=False)
CASP11_dataset = Protein_Dataset(CASP11_path, proteinList_addr, padding=False)
CASP12_dataset = Protein_Dataset(CASP12_path, proteinList_addr, padding=False)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
CASP11_loader = DataLoader(CASP11_dataset, batch_size=1, shuffle=False, num_workers=4)
CASP12_loader = DataLoader(CASP12_dataset, batch_size=1, shuffle=False, num_workers=4)

def cuda_var_wrapper(var, volatile=False):
    if use_cuda:
        var = Variable(var, volatile=volatile).cuda()
    else:
        var = Variable(var, volatile=volatile)
    return var

def evaluate(cnn, dataloader):
    cnn.eval()
    correct = 0
    total = 0
    for i, (features, labels, lengths) in enumerate(dataloader):
        features = cuda_var_wrapper(features, volatile=True)
        labels = cuda_var_wrapper(labels.view(-1), volatile=True)
        seq_len = labels.size()[0]
        output = cnn(features).view(-1, 8)
        output = F.log_softmax(output)
        _, prediction = torch.max(output, 1)
        correct_prediction = torch.sum(torch.eq(prediction, labels).data)
        correct += correct_prediction
        total += seq_len
    print("{} out of {} label predictions are correct".format(correct, total))
    return correct / total

def ensemble_evaluate(cnn_list, dataloader):
    correct = 0
    total = 0
    for i, (features, labels, lengths) in enumerate(dataloader):
        output = 0
        features = cuda_var_wrapper(features, volatile=True)
        labels = cuda_var_wrapper(labels.view(-1), volatile=True)
        seq_len = labels.size()[0]
        for cnn in cnn_list:
            inference = cnn(features).view(-1, 8)
            inference = F.log_softmax(inference)
            output += inference
        output /= len(cnn_list)
        _, prediction = torch.max(output, 1)
        correct_prediction = torch.sum(torch.eq(prediction, labels).data)
        correct += correct_prediction
        total += seq_len
    print("{} out of {} label predictions are correct".format(correct, total))
    return correct / total


cnn1 = MUFold_ss().cuda() if use_cuda else MUFold_ss()
cnn2 = MUFold_ss().cuda() if use_cuda else MUFold_ss()
cnn3 = MUFold_ss().cuda() if use_cuda else MUFold_ss()
cnn4 = MUFold_ss().cuda() if use_cuda else MUFold_ss()
cnn5 = MUFold_ss().cuda() if use_cuda else MUFold_ss()
cnn6 = MUFold_ss().cuda() if use_cuda else MUFold_ss()
cnn7 = MUFold_ss().cuda() if use_cuda else MUFold_ss()
cnn8 = MUFold_ss().cuda() if use_cuda else MUFold_ss()
cnn9 = MUFold_ss().cuda() if use_cuda else MUFold_ss()

model_path1 = os.path.join(os.getcwd(), 
    'saved_model/cnn_exp_adam_3_epochs_30_batch_64_lr_0.005_max_seq_len_698')
model_path2 = os.path.join(os.getcwd(), 
    'saved_model/cnn_exp_SGD_retrain2_epochs_30_batch_64_lr_0.001_max_seq_len_698')
model_path3 = os.path.join(os.getcwd(), 
    'saved_model/cnn_exp_SGD_retrain_epochs_15_batch_64_lr_0.0001_max_seq_len_698')
model_path4 = os.path.join(os.getcwd(), 
    'saved_model/cnn_exp3_epochs_20_batch_64_lr_0.005_max_seq_len_698')
model_path5 = os.path.join(os.getcwd(), 
    'saved_model/cnn_exp_adam_2_epochs_15_batch_64_lr_1e-05_max_seq_len_698')
model_path6 = os.path.join(os.getcwd(), 
    'saved_model/cnn_exp_new_1_epochs_20_batch_64_lr_0.001_max_seq_len_698')
model_path7 = os.path.join(os.getcwd(), 
    'saved_model/cnn_exp_new_2_epochs_20_batch_64_lr_0.005_max_seq_len_698')
model_path8 = os.path.join(os.getcwd(), 
    'saved_model/cnn_exp_new_3_epochs_20_batch_64_lr_0.005_max_seq_len_698')
model_path9 = os.path.join(os.getcwd(), 
    'saved_model/cnn_exp_new_4_epochs_20_batch_64_lr_0.001_max_seq_len_698')

cnn1.load_state_dict(torch.load(model_path1))
cnn2.load_state_dict(torch.load(model_path2))
cnn3.load_state_dict(torch.load(model_path3))
cnn4.load_state_dict(torch.load(model_path4))
cnn5.load_state_dict(torch.load(model_path5))
cnn6.load_state_dict(torch.load(model_path6))
cnn7.load_state_dict(torch.load(model_path7))
cnn8.load_state_dict(torch.load(model_path8))
cnn9.load_state_dict(torch.load(model_path9))
cnn_list = [cnn1, cnn2, cnn3, cnn4, cnn5, cnn6, cnn7, cnn8, cnn9]

def eval(cnn):
    if isinstance(cnn, list):
        print('ensemble validation')
        accuracy_test = ensemble_evaluate(cnn, test_loader)
        accuracy_CASP11 = ensemble_evaluate(cnn, CASP11_loader)
        accuracy_CASP12 = ensemble_evaluate(cnn, CASP12_loader)
    else:
        print('individual evaluation')
        accuracy_test = evaluate(cnn, test_loader)
        accuracy_CASP11 = evaluate(cnn, CASP11_loader)
        accuracy_CASP12 = evaluate(cnn, CASP12_loader)

    print("Test accuracy {:.3f}".format(accuracy_test))
    print("CASP11 accuracy {:.3f}".format(accuracy_CASP11))
    print("CASP12 accuracy {:.3f}".format(accuracy_CASP12))

for i, cnn in enumerate(cnn_list):
    print('evaluating cnn{}'.format(i+1))
    eval(cnn)
eval(cnn_list)