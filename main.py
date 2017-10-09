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

parser = argparse.ArgumentParser(description='PyTorch implementation of Mufold-ss paper')
parser.add_argument('run', metavar='DIR', help='directory to save the summary and model')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of epochs to run (default: 10)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='batch size of training data (default: 64)')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='initial learning rate (default: 1e-4)')
parser.add_argument('--max_seq_len', default=300, type=int, metavar='N', help='batch sequence length of training data (default: 300, max: 698)')
parser.add_argument('-c', '--clean', action='store_true', default=False, help="If true, clear the summary directory first")
parser.add_argument('--adjust_lr', action='store_true', default=False, help="If true, adjust lr based on validation set accuracy")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

SetOf7604Proteins_path = '../data/SetOf7604Proteins/'
CASP11_path = '../data/CASP11/'
CASP12_path = '../data/CASP12/'
trainList_addr = 'trainList'
validList_addr = 'validList'
testList_addr = 'testList'
proteinList_addr = 'proteinList'

train_dataset = Protein_Dataset(SetOf7604Proteins_path, trainList_addr, args.max_seq_len)
valid_dataset = Protein_Dataset(SetOf7604Proteins_path, validList_addr, padding=False)
test_dataset = Protein_Dataset(SetOf7604Proteins_path, testList_addr, padding=False)
CASP11_dataset = Protein_Dataset(CASP11_path, proteinList_addr, padding=False)
CASP12_dataset = Protein_Dataset(CASP12_path, proteinList_addr, padding=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
CASP11_loader = DataLoader(CASP11_dataset, batch_size=1, shuffle=False, num_workers=4)
CASP12_loader = DataLoader(CASP12_dataset, batch_size=1, shuffle=False, num_workers=4)


def cuda_var_wrapper(var):
    if use_cuda:
        var = Variable(var).cuda()
    else:
        var = Variable(var)
    return var

def evaluate(cnn, dataloader):
    cnn.eval()
    correct = 0
    total = 0
    for i, (features, labels) in enumerate(dataloader):
        features = cuda_var_wrapper(features)
        labels = cuda_var_wrapper(labels.view(-1))
        seq_len = labels.size()[0]
        output = cnn(features)
        output = output.view(-1, 8)
        _, prediction = torch.max(output, 1)
        correct_prediction = torch.sum(torch.eq(prediction, labels).data)
        # print("correct_prediction: {}; seq_len: {}".format(correct_prediction, seq_len))
        correct += correct_prediction
        total += seq_len
    print("{} out of {} label predictions are correct".format(correct, total))
    return correct / total


writer_path = os.path.join("logger", args.run)
if os.path.exists(writer_path) and args.clean:
    shutil.rmtree(writer_path)

# Save model information
save_model_dir = os.path.join(os.getcwd(), "saved_model")
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
model_name = ['cnn', str(args.run), 'epochs', str(args.epochs),
              'batch', str(args.batch_size),
              'lr', str(args.lr),
              'max_seq_len', str(args.max_seq_len)]
model_name = '_'.join(model_name)
model_path = os.path.join(save_model_dir, model_name)

writer = SummaryWriter(log_dir=writer_path)
cnn = MUFold_ss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2)

if use_cuda:
    cnn.cuda()
    criterion.cuda()

best_accuracy = 0
start = time.time()
for epoch in range(args.epochs):
    cnn.train()
    step_counter = 0
    for i, (features, labels) in enumerate(train_loader):
        features = cuda_var_wrapper(features)
        labels = cuda_var_wrapper(labels)
        output = cnn(features)
        optimizer.zero_grad()
        loss = criterion(output.view(-1, 8), labels.view(-1))
        step_counter += features.size()[0]
        writer.add_scalar('data/loss', loss.data[0], step_counter + epoch * len(train_dataset))
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
               %(epoch+1, args.epochs, i+1, len(train_dataset)//args.batch_size, loss.data[0]))

    # Validate for each epoch. Only save the best model based on validation accuracy. Adjust lr
    # if specified in argparse.
    accuracy = evaluate(cnn, valid_loader)
    if args.adjust_lr:
        scheduler.step(accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(cnn.state_dict(), model_path)
    writer.add_scalar('data/accuracy', accuracy, (epoch + 1) * len(train_dataset))
    print("Validation accuracy {:.3f}".format(accuracy))

print("Time spent on training: {:.2f}s".format(time.time() - start))
# writer.export_scalars_to_json("./all_scalars.json")
writer.close()

# Test set accuracy
accuracy_test = evaluate(cnn, test_loader)
accuracy_CASP11 = evaluate(cnn, CASP11_loader)
accuracy_CASP12 = evaluate(cnn, CASP12_loader)
print("Test accuracy {:.3f}".format(accuracy_test))
print("CASP11 accuracy {:.3f}".format(accuracy_CASP11))
print("CASP12 accuracy {:.3f}".format(accuracy_CASP12))