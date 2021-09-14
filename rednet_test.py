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
    # RedNet
    device = torch.device('cuda', 0)
    model = RedNet(depth = 50)
    model.cuda().to(device)

    # load weight
    model_dict = model.state_dict()
    # Print model's state_dict
    l1 = []
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        # print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        l1.append(model.state_dict()[param_tensor].size())

    pretrained = torch.load('rednet50.pth')

    l2 = []
    print("pretrained_dict's state_dict:")
    for param_tensor in pretrained['state_dict']:
        # print(param_tensor, "\t", pretrained['state_dict'][param_tensor].size())
        l2.append(pretrained['state_dict'][param_tensor].size())

    print()

    # 449 451 (fc.weight and fc.bias)
    # print("len(l1), len(l2)", len(l1), len(l2))
    # for idx in range(len(l1)):
    #     print(l1[idx] == l2[idx])

    pretrained_dict = pretrained['state_dict']
    # model_dict.update(pretrained_dict)
    new = list(pretrained_dict.items())
    count = 0
    for key in model_dict:
        layer_name, weights = new[count]
        model_dict[key] = weights
        count += 1
    model.load_state_dict(model_dict)
    # model.load_state_dict(torch.load('rednet50.pth'))
    model.eval()

    i = torch.randn(1, 3, 224, 224).cuda(device)
    # print(model)
    o = model(i)
    # print(o)
    print(o.shape)

    # checkpoint = model.state_dict()
    # torch.save(checkpoint, 'test.pth')

    # ## test weight
    # device = torch.device('cuda', 0)
    # model = RedNet(depth = 50)
    # model.cuda().to(device)
    # model.load_state_dict(torch.load('test.pth'))
    # model.eval()
    # i = torch.randn(1, 3, 224, 224).cuda(device)
    # o = model(i)
    # print(o)
    # print(o.shape)