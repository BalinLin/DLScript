#! /usr/bin/env python3
# coding=utf-8

import cv2, torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
# import yaml
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
import torchvision.models as models
# from involution_cuda import involution
# from involution import involution
from rednet import *
from model import *


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == "__main__":
    ## RedNet
    # device = torch.device('cuda', 0)
    # model = RedNet(depth = 50)
    # model.cuda().to(device)

    # # load weight
    # model_dict = model.state_dict()
    # # Print model's state_dict
    # l1 = []
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     # print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #     l1.append(model.state_dict()[param_tensor].size())

    # pretrained_dict = torch.load('rednet50.pth')

    # l2 = []
    # print("pretrained_dict's state_dict:")
    # for param_tensor in pretrained_dict['state_dict']:
    #     # print(param_tensor, "\t", pretrained_dict['state_dict'][param_tensor].size())
    #     l2.append(pretrained_dict['state_dict'][param_tensor].size())

    # print()

    # ## 449 451 (fc.weight and fc.bias)
    # # print("len(l1), len(l2)", len(l1), len(l2))
    # # for idx in range(len(l1)):
    # #     print(l1[idx] == l2[idx])

    # pretrained_dict = pretrained_dict['state_dict']
    # # model_dict.update(pretrained_dict)
    # new = list(pretrained_dict.items())
    # count = 0
    # for key in model_dict:
    #     layer_name, weights = new[count]
    #     model_dict[key] = weights
    #     count += 1
    # model.load_state_dict(model_dict)
    # # model.load_state_dict(torch.load('rednet50.pth'))
    # model.eval()

    # i = torch.randn(1, 3, 224, 224).cuda(device)
    # # print(model)
    # o = model(i)
    # print(o)
    # print(o.shape)

    # checkpoint = model.state_dict()
    # torch.save(checkpoint, 'test.pth')

    ### test weight
    # device = torch.device('cuda', 0)
    # model = RedNet(depth = 50)
    # model.cuda().to(device)
    # model.load_state_dict(torch.load('test.pth'))
    # model.eval()
    # i = torch.randn(1, 3, 224, 224).cuda(device)
    # o = model(i)
    # print(o)
    # print(o.shape)

    ## ResNet
    device = torch.device('cuda', 0)
    resnet50 = models.resnet50(pretrained=True)

    # for param in resnet50.parameters():
    #     param.requires_grad = True
    pretrained_weights = resnet50.conv1.weight.clone()
    resnet50.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
    # resnet50.conv1.weight[:,:3,:,:] = pretrained_weights
    # resnet50.conv1.weight[:,3,:,:].data.normal_(0.0, std=0.01)
    # print(resnet50.requires_grad)

    ## access layers https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/2
    print(resnet50.conv1.weight.is_leaf)
    print(resnet50.conv1.weight.requires_grad)
    print(resnet50.conv1.weight.size())
    print(resnet50.conv1.weight[:, :3].size())
    print(resnet50.layer4[0].conv1.weight.size())
    with torch.no_grad():  # https://discuss.pytorch.org/t/how-to-manually-set-the-weights-in-a-two-layer-linear-model/45902
        resnet50.conv1.weight[:, :3] = pretrained_weights # https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py
        resnet50.conv1.weight[:, 3] = resnet50.conv1.weight[:, 0]

    # resnet50 = ResNet50Bottom(resnet50)  # https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707
    resnet50 = nn.Sequential(*list(resnet50.children())[:-2]) # https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
    resnet50.cuda().to(device)
    # resnet50.avgpool = Identity()
    # resnet50.fc = Identity()


    # print(resnet50)
    l1 = []
    for param_tensor in resnet50.state_dict():
        # print(param_tensor, "\t", resnet50.state_dict()[param_tensor].size())
        # print(name, param.data)
        l1.append(resnet50.state_dict()[param_tensor].size())
    # summary(resnet50, (1, 3, 224, 224))
    i = torch.randn(48, 4, 224, 224).cuda(device)
    o = resnet50(i)
    # print(o)
    print(o.shape)

    ### ResNet50Bottom
    res50_model = models.resnet50(pretrained=True)
    res50_conv2 = ResNet50Bottom(res50_model)
    res50_conv2.cuda().to(device)
    i = torch.randn(48, 3, 224, 224).cuda(device)
    o = res50_conv2(i)
    print(o.shape)
    l3 = []
    for param_tensor in res50_conv2.state_dict():
        # print(param_tensor, "\t", resnet50.state_dict()[param_tensor].size())
        # print(name, param.data)
        l3.append(res50_conv2.state_dict()[param_tensor].size())
    # print(res50_conv2)


    ## ModelSpatial
    model = ModelSpatial()
    model.cuda().to(device)
    i = torch.randn(48, 3, 224, 224).cuda(device)
    o = model(i)
    print(o.shape)
    l2 = []
    # print("pretrained_dict's state_dict:")
    for param_tensor in model.state_dict():
        # print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        l2.append(model.state_dict()[param_tensor].size())

    print("len(l1), len(l2), len(l3)")
    print(len(l1), len(l2), len(l3))
    for idx in range(len(l2)):
        if not (l1[idx] == l2[idx] == l3[idx]):
            print(idx, False)
        # print(l1[idx] == l2[idx] == l3[idx])