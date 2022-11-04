
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import *
from model import model_parts


class TCNN(nn.Module):
    def __init__(self,in_channels, out_channels_res_1, out_neurons ,num_classes):

        self.in_channels = in_channels
        self.out_channels_res_1 = out_channels_res_1
        self.conv_out = nn.Conv1d(out_channels_res_1,out_neurons,kernel_size=1, padding=1)
        self.fc_out = nn.Linear(out_neurons,num_classes)
        self.res_block = res_block(in_channels,out_channels_res_1)

    def forward(self, x):
        x = res_block(x)
        x

