import torch
import torch.nn as nn
from torch.autograd import Variable


class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv,self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=2, padding=1),
            nn.BatchNorm1d(),
            nn.ReLU(),

            nn.Conv1d(in_channels, out_channels, kernel_size=2, padding=1),
            nn.BatchNorm1d(),
            nn.ReLU()

        )
    def forward(self, x):
        return self.double_conv(x)

class res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(res_block, self).__init__()

        self.double_conv = double_conv(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        res = x
        out = self.double_conv(x)
        out += res
        return(out)

