# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn
import torchvision.models as models

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

    
class block(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, identity_downsample = None, stride = 1):
        super(block, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, stride, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel*self.expansion, 1, 1, 0),
            nn.BatchNorm2d(out_channel*self.expansion),
        )
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    def forward(self, x):
        identity = x
        x = self.seq(x)
        if self.identity_downsample:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
        
class myResnet(nn.Module):
    def __init__(self, block, layers, num_class):
        super(myResnet, self).__init__()
        self.in_channel = 64
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        #ResNet Layers
        self.layer1 = self.make_layer(block, layers[0], 64, 1)
        self.layer2 = self.make_layer(block, layers[1], 128, 2)
        self.layer3 = self.make_layer(block, layers[2], 256, 2)
        self.layer4 = self.make_layer(block, layers[3], 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_class)


    def make_layer(self, block, num_residual_block, out_channel, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channel != out_channel * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel*block.expansion, 1, stride),
                nn.BatchNorm2d(out_channel*block.expansion),
            )
        layers.append(block(self.in_channel, out_channel, identity_downsample, stride))
        self.in_channel = out_channel * block.expansion

        for _ in range(1, num_residual_block):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.seq(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNext50(num_class , device):
    model = models.resnext50_32x4d(pretrained=True).to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)
    return model