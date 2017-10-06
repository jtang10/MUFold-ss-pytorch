from __future__ import print_function, division

import time
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_loading import Protein_Dataset

parser = argparse.ArgumentParser(description='PyTorch implementation of Mufold-ss paper')
parser.add_argument('run', metavar='DIR', help='directory to save the summary and model')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='batch size of training data')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--max_seq_len', default=300, type=int, metavar='N', help='cutoff sequence length of training data')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

SetOf7604Proteins_path = '../data/SetOf7604Proteins/'
trainList_addr = 'trainList'
validList_addr = 'validList'
testList_addr = 'testList'

train_dataset = Protein_Dataset(SetOf7604Proteins_path, trainList_addr, args.max_seq_len)
valid_dataset = Protein_Dataset(SetOf7604Proteins_path, validList_addr, padding=False)
test_dataset = Protein_Dataset(SetOf7604Proteins_path, testList_addr, padding=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

def cuda_var_wrapper(var):
    if use_cuda:
        var = Variable(var).cuda()
    else:
        var = Variable(var)
    return var

class BasicConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, **kwargs).double()
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x.float())
        return F.relu(x, inplace=True)

class basic_inception_module(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(basic_inception_module, self).__init__()
        self.conv_1x1s = nn.ModuleList(
            [BasicConv1d(in_channels, 100, kernel_size=1) for i in range(3)])
        self.conv_3x3s = nn.ModuleList(
            [BasicConv1d(100, 100, kernel_size=3, padding=1) for i in range(4)])

    def forward(self, x):
        branch1 = self.conv_1x1s[0](x).double()
        branch2 = self.conv_1x1s[1](x).double()
        branch2 = self.conv_3x3s[0](branch2).double()
        branch3 = self.conv_1x1s[2](x).double()
        branch3 = self.conv_3x3s[1](branch3).double()
        branch3 = self.conv_3x3s[2](branch3).double()
        branch3 = self.conv_3x3s[3](branch3).double()

        return torch.cat([branch1, branch2, branch3], 1)

class Deep3I(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Deep3I, self).__init__()
        self.input_layer = basic_inception_module(in_channels)
        self.intermediate_layer = basic_inception_module(300)
        self.dropout = nn.Dropout(0.4)
        

    def forward(self, x):
        branch1 = self.input_layer(x)
        branch2 = self.intermediate_layer(self.input_layer(x))
        branch3 = self.input_layer(x)
        for i in range(3):
            branch3 = self.intermediate_layer(branch3)
        output = torch.cat([branch1, branch2, branch3], 1)
        output = self.dropout(output)
        return output

class MUFold_ss(nn.Module):
    def __init__(self):
        super(MUFold_ss, self).__init__()
        self.layer1 = Deep3I(66)
        self.layer2 = Deep3I(900)
        self.fc = nn.Sequential(nn.Linear(900, 400), nn.Linear(400, 8))

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output).float()
        output = self.fc(output.permute(0, 2, 1))

        return output

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

def train():
    writer = SummaryWriter(log_dir=os.path.join("logger", args.run))
    cnn = MUFold_ss()
    if use_cuda:
        start = time.time()
        cnn = cnn.cuda()
        print("Spent {:.2f}s to load GPU model".format(time.time() - start))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    start = time.time()
    for epoch in range(epochs):
        cnn.train()
        print("Epoch {} out of {}".format(epoch + 1, epochs))
        for i, (features, labels) in enumerate(train_loader):
            features = cuda_var_wrapper(features)
            labels = cuda_var_wrapper(labels)
            output = cnn(features)
            # print(output.size()[1])
            # print(type(output.size()))
            # print(type(output.data))
            # break
            optimizer.zero_grad()
            loss = criterion(output.view(-1, 8), labels.view(-1))
            writer.add_scalar('data/loss', loss.data[0], i + epoch * len(train_dataset)//args.batch_size)
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, epochs, i+1, len(train_dataset)//args.batch_size, loss.data[0]))

        accuracy = evaluate(cnn, valid_loader)
        writer.add_scalar('data/accuracy', accuracy, epoch * len(train_dataset)//args.batch_size)
        print("Validation accuracy {:.3f}".format(accuracy))
    print("Time spent: {:.2f}s".format(time.time() - start))
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    # Test set accuracy
    accuracy_test = evaluate(cnn, test_loader)
    print("Test accuracy {:.3f}".format(accuracy_test))

    # Save the model
    save_model_dir = os.path.join(os.getcwd(), "saved_model")
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    model_counter = 0
    model_name = ['cnn', str(model_counter), 'epochs', str(args.epochs), ]
    torch.save(the_model.state_dict(), )

if __name__ == '__main__':
    train()