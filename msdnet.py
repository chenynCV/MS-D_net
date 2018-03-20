import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from IPython import embed

__all__ = ['msdn']

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Dilate_conv_block(nn.Module):
    def __init__(self, in_ch=1, filter_size=3, w=2, i=1):
        super(Dilate_conv_block, self).__init__()

        self.w = w
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        layers = []    
        # Add N layers
        for j in range(w):
            s = (i*w + j)%10 + 1
            layers.append(self._dilate_conv(in_ch=in_ch, out_ch=1, filter_size=filter_size, dilate=s))
        # Add to Module List
        self.layers = nn.ModuleList(layers)

    def _dilate_conv(self, in_ch, out_ch, filter_size, dilate):
        conv = nn.Conv2d(in_ch, out_ch, filter_size, padding=dilate, dilation=dilate)
        return nn.Sequential(conv)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)

        out = []
        for j in range(self.w):
            out.append(self.layers[j](x))
        out = torch.cat(out, 1)
        return out
  

class MSDN_block(nn.Module):
    def __init__(self, block, in_ch=1, out_ch=3, w=1, d=50):
        super(MSDN_block, self).__init__()

        self.d = d
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(in_ch + w*d)
        self.conv2 = nn.Conv2d(in_ch + w*d, out_ch, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        current_in_channels = in_ch   
        # Add N layers
        for i in range(d):
            layers.append(block(in_ch=current_in_channels, w=w, i=i))
            current_in_channels += w
        # Add to Module List
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        prev_features = [x]
        for i in range(self.d):           
            x = self.layers[i](x)
            # Append output into previous features
            prev_features.append(x)
            x = torch.cat(prev_features, 1)

        out = self.bn2(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out, x


class MSDNet(nn.Module):
    """
    Paper: A mixed-scale dense convolutional neural network for image analysis
    Published: PNAS, Jan. 2018 
    Paper: http://www.pnas.org/content/early/2017/12/21/1715832114
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m, m.weight.data)

    def __init__(self, block, im_ch=3, w=1, d=20, stage=3, out_channels=16):
        super(MSDNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 16
        self.stage = stage

        self.conv1 = nn.Conv2d(im_ch, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)

        self.scale = nn.AvgPool2d(4, stride=4)
        self.conv1_1 = nn.Conv2d(im_ch, self.num_feats*block.expansion, kernel_size=1, bias=True)

        layers, score_ = [], []
        current_in_channels = self.num_feats*block.expansion   
        # Add N layers
        for i in range(stage):
            layers.append(MSDN_block(Dilate_conv_block, in_ch=current_in_channels, out_ch=out_channels, w=w, d=d))
            current_in_channels += w*d
            if i < stage-1:
                score_.append(nn.Conv2d(out_channels, current_in_channels, kernel_size=1, bias=True))
        # Add to Module List
        self.layers = nn.ModuleList(layers)
        self.score_ = nn.ModuleList(score_)
        
        self.apply(self.weight_init)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        x_in = x

        x_1 = self.scale(x_in)
        x_1 = self.conv1_1(x_1)

        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x) 

        x = x_1 + x

        for i in range(self.stage):           
            score, x = self.layers[i](x)
            out.append(score) 
            if i < self.stage-1:
                score_ = self.score_[i](score)
                x = x + score_
        return out


def msdn(**kwargs):
    model = MSDNet(block=Bottleneck, out_channels=kwargs['num_classes'])
    return model