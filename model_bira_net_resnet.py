import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
import torch.nn.functional as F

class Lambda(nn.Module):
	def __init__(self, lambd):
		super(Lambda, self).__init__()
		self.lambd = lambd
	def forward(self, x):
		return self.lambd(x)


class Attention_biranet(nn.Module):
    def __init__(self,last_channel_dim):
        super().__init__()
        self.last_channel_dim = last_channel_dim
        self.attention = nn.Sequential(
        nn.BatchNorm2d(self.last_channel_dim),
        nn.Conv2d(self.last_channel_dim,64,kernel_size=1,padding=0),
        nn.ReLU(),
        nn.Conv2d(64,16,kernel_size=1,padding=0),
        nn.ReLU(),
        nn.Conv2d(16,8,kernel_size=1,padding=0),
        nn.ReLU(),
        nn.Conv2d(8,1,kernel_size=1,padding=0),
        nn.Sigmoid()
        )
    def forward(self,x):
        return self.attention(x)

class BiraNet_half(nn.Module):
    def __init__(self,last_channel_dim=512):
        super().__init__()
        resNet = models.resnet18(pretrained=True)
        resNet = list(resNet.children())[:-2]
        self.last_channel_dim = last_channel_dim
        self.features = nn.Sequential(*resNet)
        self.attention = Attention_biranet(self.last_channel_dim)

        self.up_c2 = nn.Conv2d(1,self.last_channel_dim, kernel_size = 1, padding = 0,bias = False)
        nn.init.constant_(self.up_c2.weight, 1)

        self.net_b = nn.Conv2d(self.last_channel_dim, self.last_channel_dim, kernel_size=1, padding=1)
        self.net_b_relu = nn.ReLU()

        # self.denses = nn.Sequential(
        # nn.Linear(self.last_channel_dim,256),
        # nn.Dropout(0.5),
        # nn.Linear(256, classes_num)
        # )
    def forward(self,x):
        x_1 = self.features(x)
        #print('featur extractor output shape : {}'.format(x.shape))
        atten_layers = self.attention(x_1)
        #print('attention mask output shape : {}'.format(atten_layers.shape))
        atten_layers = self.up_c2(atten_layers)
        #print mask_features.shape
        mask_features = torch.mul(atten_layers,x_1)
        #print('masked features output shape : {}'.format(mask_features.shape))
        #print('kernel size: {}'.format(mask_features.size()[2:]))
        gap_features = F.avg_pool2d(mask_features,kernel_size=mask_features.size()[2:])
        #print('gap features output shape : {}'.format(gap_features.shape))
        #print('kernel size: {}'.format(atten_layers.size()[2:]))
        gap_mask = F.avg_pool2d(atten_layers,kernel_size=atten_layers.size()[2:])
        #print('gap mask output shape : {}'.format(gap_mask.shape))
        gap =  torch.squeeze(Lambda(lambda x: x[0]/x[1])([gap_features, gap_mask]))
        #print('gap output shape : {}'.format(gap.shape))
        g = self.net_b_relu(self.net_b(x_1))
        g = F.avg_pool2d(g, kernel_size=g.size()[2:]).squeeze()
        pre_final = (gap + g) / 2
        if len(pre_final.size()) == 1:
            pre_final = pre_final.unsqueeze(0)
        return pre_final

class BiraNet_ResNet(nn.Module):
    def __init__(self,last_channel_dim,classes_num):
        super(BiraNet_ResNet, self).__init__()
        self.biranet_half_1 = BiraNet_half(last_channel_dim)
        self.biranet_half_2 = BiraNet_half(last_channel_dim)
        self.denses = nn.Sequential(
        nn.Linear(last_channel_dim,256),
        nn.Dropout(0.5),
        nn.Linear(256, classes_num)
        )

    def forward(self, x):
        pre_final_1 = self.biranet_half_1(x)
        pre_final_2 = self.biranet_half_2(x)
        res = torch.bmm(pre_final_1.unsqueeze(2), pre_final_2.unsqueeze(1))
        # print(res.shape)
        res = torch.bmm(res, res.permute(0, 2, 1))
        # print(res.shape)
        res = torch.mean(res, axis=1)
        # print(res.shape)

        #### Contrastive Prototype Learning part

        out = self.denses(res)

        return out


# model = BiraNet_ResNet(512,4)

# # print(model)

# print(sum([param.numel() for param in model.parameters()]))

# # print(model.features_1)

# inp = torch.rand(2,3,1316,2632)

# out = model(inp)

# print(out.shape)

# resNet_1 = models.resnet50(pretrained=False)
# resNet_1 = list(resNet_1.children())[:-4]
# print(resNet_1)