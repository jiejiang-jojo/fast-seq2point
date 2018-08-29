import sys
import math
import inspect
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

    elif layer.weight.ndimension() == 3:
        (n_out, n_in, width) = layer.weight.size()
        n = n_in * width

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
    
    seq_len = 41
    print('seq_len: ', seq_len)
    
    def __init__(self):
        
        super(CNN3, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(21, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(21, 1), stride=(1, 1), padding=(0, 0), bias=True)
        
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
    
    seq_len = 253
    
    def __init__(self):
        
        super(CNN7, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(43, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(43, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(43, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(43, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(43, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(43, 1), stride=(1, 1), padding=(0, 0), bias=True)
        
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


class WaveNet(nn.Module):
    
    seq_len = 64
    print('seq_len: ', seq_len)
    
    def __init__(self):
        
        super(WaveNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), dilation=(2, 1), bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), dilation=(4, 1), bias=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), dilation=(8, 1), bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), dilation=(16, 1), bias=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), dilation=(32, 1), bias=True)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.conv5)
        init_layer(self.conv6)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        

    def forward(self, input):
        
        x = input
        x = x.view(x.shape[0], 1, x.shape[1], 1)
  
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        x = x.view(x.shape[0], x.shape[2])

        return x


class DilatedResidualBlock(nn.Module):
    def __init__(self, out_channels, skip_channels, kernel_size, dilation, bias):
        super(DilatedResidualBlock, self).__init__()
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.dilated_conv = nn.Conv1d(out_channels, 2 * out_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation, bias=bias)
        self.mixing_conv = nn.Conv1d(out_channels, out_channels + skip_channels, kernel_size=1, bias=False)
        self.init_weights()

    def init_weights(self):
        init_layer(self.dilated_conv)
        init_layer(self.mixing_conv)

    def forward(self, data_in):
        
        out = self.dilated_conv(data_in)
        out1 = out.narrow(-2, 0, self.out_channels)
        out2 = out.narrow(-2, self.out_channels, self.out_channels)
        tanh_out = F.tanh(out1)
        sigm_out = F.sigmoid(out2)
        data = F.mul(tanh_out, sigm_out)
        data = self.mixing_conv(data)
        res = data.narrow(-2, 0, self.out_channels)
        skip = data.narrow(-2, self.out_channels, self.skip_channels)
        res = res + data_in
        return res, skip


class WaveNet2(nn.Module):

    layers = 6
    kernel_size = 3 # has to be odd integer, since even integer may break dilated conv output size
    seq_len = (2 ** layers - 1) * (kernel_size - 1) + 1
    print('seq_len: ', seq_len)

    def __init__(self):
        super(WaveNet2, self).__init__()
        channels = 32
        skip_channels = 8

        self.causal_conv = nn.Conv1d(1, channels, kernel_size=1, bias=False)
        self.blocks = [DilatedResidualBlock(channels, skip_channels, WaveNet2.kernel_size, 2**i, True)
                       for i in range(WaveNet2.layers)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"dilatedConv{i}", block)
        self.penultimate_conv = nn.Conv1d(skip_channels, 64, kernel_size=WaveNet2.kernel_size, padding=(WaveNet2.kernel_size-1)//2, bias=True)
        self.final_conv = nn.Conv1d(64, 1, kernel_size=WaveNet2.kernel_size, padding=(WaveNet2.kernel_size-1)//2, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.causal_conv)
        init_layer(self.penultimate_conv)
        init_layer(self.final_conv)

    def forward(self, data_in):
        data_in = data_in.view(data_in.shape[0], 1, data_in.shape[1])
        data_out = self.causal_conv(data_in)
        skip_connections = []
        for block in self.blocks:
            data_out, skip_out = block(data_out)
            skip_connections.append(skip_out)
        skip_out = skip_connections[0]
        for skip_other in skip_connections[1:]:
            skip_out = skip_out + skip_other
        data_out = F.relu(skip_out)
        data_out = self.penultimate_conv(data_out)
        data_out = self.final_conv(data_out)
        data_out = data_out.narrow(-1, WaveNet2.seq_len//2, data_out.size()[-1]-WaveNet2.seq_len+1)
        return data_out.view(data_out.shape[0], data_out.shape[2])



class BGRU(nn.Module):
    
    seq_len = 41
    print('seq_len: ', seq_len)
    
    def __init__(self):
        
        super(BGRU, self).__init__()
        
        self.bgru = nn.GRU(input_size=1, hidden_size=64, num_layers=2, bias=True, batch_first=True, dropout=0., bidirectional=True)
        
        self.fc_final = nn.Linear(128, 1)

        self.init_weights()
        
    def _init_param(self, param):
        
        if param.ndimension() == 1:
            param.data.fill_(0.)
            
        elif param.ndimension() == 2:
            n = param.size(-1)
            std = math.sqrt(2. / n)
            scale = std * math.sqrt(3.)
            param.data.uniform_(-scale, scale)
        
    def init_weights(self):
        
        for param in self.bgru.parameters():
            self._init_param(param)
            
        init_layer(self.fc_final)

    def forward(self, input):
        
        x = input
        x = x.view(x.shape[0], x.shape[1], 1)
        '''(batch_size, time_steps, 1)'''
        
        (x, h) = self.bgru(x)
        '''x: (batch_size, time_steps, feature_maps)'''
        
        x = self.fc_final(x)
        '''(batch_size, time_steps, 1)'''
        
        x = x.view(x.shape[0 : 2])
        '''(batch_size, time_steps)'''
        
        seq_len = BGRU.seq_len
        width = x.shape[1] - seq_len + 1
        output = x[:, seq_len // 2 : seq_len // 2 + width]
        '''(batch_size, width)'''
        
        return output
        

class WaveNetBGRU(nn.Module):
    
    layers = 6
    kernel_size = 3 # has to be odd integer, since even integer may break dilated conv output size
    seq_len = (2 ** layers - 1) * (kernel_size - 1) + 1
    print('seq_len: ', seq_len)

    def __init__(self):
        super(WaveNetBGRU, self).__init__()
        channels = 32
        skip_channels = 8

        self.causal_conv = nn.Conv1d(1, channels, kernel_size=1, bias=False)
        self.blocks = [DilatedResidualBlock(channels, skip_channels, WaveNetBGRU.kernel_size, 2**i, True)
                       for i in range(WaveNetBGRU.layers)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"dilatedConv{i}", block)
        self.penultimate_conv = nn.Conv1d(skip_channels, 64, kernel_size=WaveNetBGRU.kernel_size, padding=(WaveNetBGRU.kernel_size-1)//2, bias=True)
        self.final_conv = nn.Conv1d(64, 1, kernel_size=WaveNetBGRU.kernel_size, padding=(WaveNetBGRU.kernel_size-1)//2, bias=True)
    
        self.bgru = nn.GRU(input_size=64, hidden_size=64, num_layers=2, bias=True, batch_first=True, dropout=0., bidirectional=True)
        
        self.fc_final = nn.Linear(128, 1)
        
        self.init_weights()

    def _init_param(self, param):
        
        if param.ndimension() == 1:
            param.data.fill_(0.)
            
        elif param.ndimension() == 2:
            n = param.size(-1)
            std = math.sqrt(2. / n)
            scale = std * math.sqrt(3.)
            param.data.uniform_(-scale, scale)

    def init_weights(self):
        init_layer(self.causal_conv)
        init_layer(self.penultimate_conv)
        
        for param in self.bgru.parameters():
            self._init_param(param)
            
        init_layer(self.fc_final)
        

    def forward(self, data_in):
        data_in = data_in.view(data_in.shape[0], 1, data_in.shape[1])
        
        data_out = self.causal_conv(data_in)
        skip_connections = []
        for block in self.blocks:
            data_out, skip_out = block(data_out)
            skip_connections.append(skip_out)
        skip_out = skip_connections[0]
        for skip_other in skip_connections[1:]:
            skip_out = skip_out + skip_other
        data_out = F.relu(skip_out)
        data_out = self.penultimate_conv(data_out)
        '''(batch_size, feature_maps, time_steps)'''
        
        data_out = data_out.transpose(1, 2)
        '''(batch_size, time_steps, feature_maps)'''
        
        (data_out, h) = self.bgru(data_out)
        
        data_out = self.fc_final(data_out)
        '''(batch_size, time_steps, 1)'''
        
        data_out = data_out.view(data_out.shape[0 : 2])
        '''(batch_size, time_steps)'''
        
        seq_len = WaveNetBGRU.seq_len
        width = data_out.shape[1] - seq_len + 1
        output = data_out[:, seq_len // 2 : seq_len // 2 + width]
        '''(batch_size, width)'''
        
        return output
        # return data_out.view(data_out.shape[0], data_out.shape[2])
        
        
MODELS = {cname: cls
          for cname, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
          if issubclass(cls, nn.Module)}