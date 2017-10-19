from __future__ import print_function, division

import time
import argparse
import os
import datetime
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
parser.add_argument('--run', default='exp_test', metavar='DIR', help='directory to save the summary and model')
parser.add_argument('-e', '--epochs', default=15, type=int, metavar='N', help='number of epochs to run (default: 15)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='batch size of training data (default: 64)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 1e-3)')
parser.add_argument('--min_lr', default=1e-6, type=float, metavar='LR', help='minimum learning rate for lr scheduling (default: 1e-6)')
parser.add_argument('--patience', default=3, type=int, metavar='N', help='patience for lr scheduling (default: 3)')
parser.add_argument('--max_seq_len', default=698, type=int, metavar='N', help='batch sequence length of training data (default: 698, max: 698)')
parser.add_argument('--dropout', default=0.4, type=float, metavar='N', help='dropout factor. default: 0.4')
parser.add_argument('--adjust_lr', action='store_true', default=False, help="If specified, adjust lr based on validation set accuracy")
parser.add_argument('-c', '--clean', action='store_true', default=False, help="If specified, clear the summary directory first")
parser.add_argument('--reload', action='store_true', default=False, help="If specified, retrain a saved model")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

SetOf7604Proteins_path = os.path.expanduser('~/bio/data/SetOf7604Proteins/')
CASP11_path = os.path.expanduser('~/bio/data/CASP11/')
CASP12_path = os.path.expanduser('~/bio/data/CASP12/')
trainList_addr = 'trainList'
validList_addr = 'validList'
testList_addr = 'testList'
proteinList_addr = 'proteinList'

train_dataset = Protein_Dataset(SetOf7604Proteins_path, trainList_addr, args.max_seq_len)
valid_dataset = Protein_Dataset(SetOf7604Proteins_path, validList_addr, args.max_seq_len)
test_dataset = Protein_Dataset(SetOf7604Proteins_path, testList_addr, args.max_seq_len)
CASP11_dataset = Protein_Dataset(CASP11_path, proteinList_addr, padding=False)
CASP12_dataset = Protein_Dataset(CASP12_path, proteinList_addr, padding=False)
len_train_dataset = len(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
CASP11_loader = DataLoader(CASP11_dataset, batch_size=1, shuffle=False, num_workers=4)
CASP12_loader = DataLoader(CASP12_dataset, batch_size=1, shuffle=False, num_workers=4)

writer_path = os.path.join("logger", args.run)
if os.path.exists(writer_path) and args.clean:
    shutil.rmtree(writer_path, ignore_errors=True)

# Save model information
save_model_dir = os.path.join(os.getcwd(), "saved_model")
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
model_name = ['cnn', datetime.datetime.now().strftime("%b%d_%H:%M"),
              str(args.run), 'epochs', str(args.epochs), 'lr', str(args.lr)]
model_name = '_'.join(model_name)
model_path = os.path.join(save_model_dir, model_name)

def cuda_var_wrapper(var, volatile=False):
    if use_cuda:
        var = Variable(var, volatile=volatile).cuda()
    else:
        var = Variable(var, volatile=volatile)
    return var

def get_batch_accuracy(labels, output, lengths):
    correct = 0.0
    total = 0.0
    _, prediction = torch.max(output, 2)
    correct_matrix = torch.eq(prediction, labels).data
    for j, length in enumerate(lengths):
            correct += torch.sum(correct_matrix[j, :length])
    total += sum(lengths)
    return correct, total

def evaluate(cnn, dataloader):
    cnn.eval()
    correct = 0.0
    total = 0.0
    for i, (features, labels, lengths) in enumerate(dataloader):
        max_length = max(lengths)
        features = cuda_var_wrapper(features[:, :, :max_length], volatile=True)
        labels = cuda_var_wrapper(labels[:, :max_length], volatile=True)
        output = cnn(features)
        correct_batch, total_batch = get_batch_accuracy(labels, output, lengths)
        correct += correct_batch
        total += total_batch
    print("{} out of {} label predictions are correct".format(correct, total))
    return correct / total


writer = SummaryWriter(log_dir=writer_path)
cnn = MUFold_ss(dropout=args.dropout)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
# optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.patience,
                              factor=0.1, min_lr=args.min_lr, verbose=True)

if use_cuda:
    cnn.cuda()
    criterion.cuda()

load_trained_model = os.path.join(os.getcwd(), 'saved_model/cnn_exp3_epochs_20_batch_64_lr_0.005_max_seq_len_698')
if args.reload:
    cnn.load_state_dict(torch.load(load_trained_model))

best_accuracy = 0
start = time.time()
for epoch in range(args.epochs):
    cnn.train()
    step_counter = 0

    for i, (features, labels, lengths) in enumerate(train_loader):
        max_length = max(lengths)
        features = cuda_var_wrapper(features[:, :, :max_length])
        labels = cuda_var_wrapper(labels[:, :max_length])
        output = cnn(features)
        optimizer.zero_grad()
        loss = criterion(output.view(-1, 8), labels.view(-1))
        step_counter += features.size()[0]
        writer.add_scalar('data/loss', loss.data[0], step_counter + epoch * len_train_dataset)
        loss.backward()
        optimizer.step()
        if (i) % 20 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
               %(epoch+1, args.epochs, i, len_train_dataset//args.batch_size, loss.data[0]))

    accuracy_train = evaluate(cnn, train_loader)
    accuracy_valid = evaluate(cnn, valid_loader)
    if args.adjust_lr:
        scheduler.step(accuracy_valid)
    if accuracy_valid > best_accuracy:
        best_accuracy = accuracy_valid
        torch.save(cnn.state_dict(), model_path)
    writer.add_scalars('data/accuracy_group', {'accuracy_train': accuracy_train,
                                               'accuracy_valid': accuracy_valid}, (epoch + 1) * 5070)
    print("Training accuracy {:.4f}; Validation accuracy {:.4f}".format(accuracy_train, accuracy_valid))

print("Time spent on training: {:.2f}s".format(time.time() - start))
# writer.export_scalars_to_json("./all_scalars.json")
writer.close()

# Test set accuracy. Restore the best saved model instead of use the trained.
cnn.load_state_dict(torch.load(model_path))
accuracy_test = evaluate(cnn, test_loader)
accuracy_CASP11 = evaluate(cnn, CASP11_loader)
accuracy_CASP12 = evaluate(cnn, CASP12_loader)
print("Test accuracy {:.3f}".format(accuracy_test))
print("CASP11 accuracy {:.3f}".format(accuracy_CASP11))
print("CASP12 accuracy {:.3f}".format(accuracy_CASP12))