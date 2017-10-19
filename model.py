from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        for i in range(1, 4):
            branch3 = self.conv_3x3s[i](branch3).double()

        return torch.cat([branch1, branch2, branch3], 1)

class Deep3I(nn.Module):
    def __init__(self, in_channels, dropout, **kwargs):
        super(Deep3I, self).__init__()
        self.input_layers = nn.ModuleList(
            [basic_inception_module(in_channels) for i in range(3)])
        self.intermediate_layers = nn.ModuleList(
            [basic_inception_module(300) for i in range(4)])
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        branch1 = self.input_layers[0](x)
        branch2 = self.intermediate_layers[0](self.input_layers[1](x))
        branch3 = self.input_layers[2](x)
        for i in range(1, 4):
            branch3 = self.intermediate_layers[i](branch3)
        output = torch.cat([branch1, branch2, branch3], 1)
        output = self.dropout(output)
        return output

class MUFold_ss(nn.Module):
    def __init__(self, dropout=0.4):
        super(MUFold_ss, self).__init__()
        self.layer1 = Deep3I(66, dropout)
        self.layer2 = Deep3I(900, dropout)
        self.fc = nn.Sequential(nn.Linear(900, 400), nn.Linear(400, 8))

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output).float()
        output = self.fc(output.permute(0, 2, 1))

        return output