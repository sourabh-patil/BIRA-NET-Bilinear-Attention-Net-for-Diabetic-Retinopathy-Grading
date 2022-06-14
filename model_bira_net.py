import os
from tkinter import N
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

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,dilation):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True,dilation=dilation),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True,dilation=dilation),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class CNN_model_dilated_conv(nn.Module):
    def __init__(self,n_channels, n_st ,n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        
        self.conv1 = conv_block(n_channels, n_st*2,dilation=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = conv_block(n_st*2, n_st*4,dilation=2)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = conv_block(n_st*4, n_st*8,dilation=3)
        #self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = conv_block(n_st*8, n_st*16,dilation=1)
        self.dropout2d = nn.Dropout2d(0.1)
    
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.dropout2d(self.maxpool(self.conv2(x)))
        x = self.dropout2d(self.maxpool(self.conv3(x)))
        x = self.dropout2d(self.conv4(x))
        return x 


class Bira_Net_dilated_conv(nn.Module):

    def __init__(self,num_channels, n_st, n_classes):
        super(Bira_Net_dilated_conv, self).__init__()
        self.dilated_conv_feature_extractor_1 = CNN_model_dilated_conv(n_channels=num_channels, n_st=n_st, n_classes=n_classes)
        self.dilated_conv_feature_extractor_2 = CNN_model_dilated_conv(n_channels=num_channels, n_st=n_st, n_classes=n_classes)

        self.attention_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64,16,kernel_size=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(16,8,kernel_size=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(8,1,kernel_size=1,padding=0),
            nn.Sigmoid()
            )
        
        self.attention_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64,16,kernel_size=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(16,8,kernel_size=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(8,1,kernel_size=1,padding=0),
            nn.Sigmoid()
            )

        self.up_c2_1 = nn.Conv2d(1,64, kernel_size = 1, padding = 0,bias = False)
        nn.init.constant_(self.up_c2_1.weight, 1)

        self.up_c2_2 = nn.Conv2d(1,64, kernel_size = 1, padding = 0,bias = False)
        nn.init.constant_(self.up_c2_2.weight, 1)


        self.net_b_1 = nn.Conv2d(64, 64, kernel_size=1, padding=1)
        self.net_b_1_relu = nn.ReLU()
        self.net_b_2 = nn.Conv2d(64, 64, kernel_size=1, padding=1)
        self.net_b_2_relu = nn.ReLU()

        self.denses = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
            )
		
    def forward(self, x):
        x_1 = self.dilated_conv_feature_extractor_1(x)
        x_2 = self.dilated_conv_feature_extractor_2(x)
        #print('featur extractor output shape : {}'.format(x.shape))

        atten_layers_1 = self.attention_1(x_1)
        atten_layers_2 = self.attention_2(x_2)
        #print('attention mask output shape : {}'.format(atten_layers.shape))
        
        atten_layers_1 = self.up_c2_1(atten_layers_1)
        atten_layers_2 = self.up_c2_2(atten_layers_2)
        #print('attention up channel output shape : {}'.format(atten_layers.shape))

        # print(atten_layers_1.shape)
        # print(x_1.shape)
        # print(atten_layers_2.shape)
        # print(x_2.shape)
        
        mask_features_1 = torch.mul(atten_layers_1,x_1)
        mask_features_2 = torch.mul(atten_layers_2,x_2)
        #print('masked features output shape : {}'.format(mask_features.shape))
        #print('kernel size: {}'.format(mask_features.size()[2:]))
        
        gap_features_1 = F.avg_pool2d(mask_features_1,kernel_size=mask_features_1.size()[2:])
        gap_features_2 = F.avg_pool2d(mask_features_2,kernel_size=mask_features_2.size()[2:])
        #print('gap features output shape : {}'.format(gap_features.shape))
        #print('kernel size: {}'.format(atten_layers.size()[2:]))
        
        gap_mask_1 = F.avg_pool2d(atten_layers_1,kernel_size=atten_layers_1.size()[2:])
        gap_mask_2 = F.avg_pool2d(atten_layers_2,kernel_size=atten_layers_2.size()[2:])
        #print('gap mask output shape : {}'.format(gap_mask.shape))
        
        gap_1 =  torch.squeeze(Lambda(lambda x: x[0]/x[1])([gap_features_1, gap_mask_1]))
        gap_2 =  torch.squeeze(Lambda(lambda x: x[0]/x[1])([gap_features_2, gap_mask_2]))
        #print('gap output shape : {}'.format(gap.shape))

        g_1 = self.net_b_1_relu(self.net_b_1(x_1))
        g_2 = self.net_b_2_relu(self.net_b_2(x_2))
        
        g_1 = F.avg_pool2d(g_1, kernel_size=g_1.size()[2:]).squeeze()
        g_2 = F.avg_pool2d(g_2, kernel_size=g_2.size()[2:]).squeeze()

        pre_final_1 = (gap_1 + g_1) / 2
        pre_final_2 = (gap_2 + g_2) / 2  

        res = torch.bmm(pre_final_1.unsqueeze(2), pre_final_2.unsqueeze(1))
        #print(res.shape)
        res = torch.bmm(res, res.permute(0, 2, 1))
        #print(res.shape)
        res = torch.mean(res, axis=1)
        # print(res.shape)

        out = self.denses(res)
        # print('dense layer output shape : {}'.format(x.shape))
        # x1 = x
        # x = F.log_softmax(x,dim=1)
        return out

# model = Bira_Net_dilated_conv(num_channels=1,n_st=4,n_classes=4)

# print(sum([param.numel() for param in model.parameters()]))


# for param in model.up_c2_1.parameters():
#     param.requires_grad = False

# for param in model.up_c2_2.parameters():
#     param.requires_grad = False

# print(sum([param.numel() for param in model.parameters()]))

# inp = torch.rand(2,1,1316,2632)

# out= model(inp)

# print(out.shape)
# print(out1)
# print(out2.shape)
# print(out2)
