import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
        
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


class CNN3(nn.Module):
    
    seq_len = 21
    
    def __init__(self):
        
        super(CNN3, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(11, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(11, 1), stride=(1, 1), padding=(0, 0), bias=True)
        
        self.conv_final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv_final)

    def forward(self, input):
        
        x = input
        x = x.view(x.shape[0], 1, x.shape[1], 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.conv_final(x)
        x = x.view(x.shape[0], x.shape[2])
        
        return x

        
class CNN7(nn.Module):
    
    seq_len = 601
    
    def __init__(self):
        
        super(CNN7, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(101, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(101, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(101, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(101, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(101, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(101, 1), stride=(1, 1), padding=(0, 0), bias=True)
        
        self.conv_final = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.conv5)
        init_layer(self.conv6)
        init_layer(self.conv_final)

    def forward(self, input):
        
        x = input
        x = x.view(x.shape[0], 1, x.shape[1], 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.conv_final(x)
        x = x.view(x.shape[0], x.shape[2])
        
        return x