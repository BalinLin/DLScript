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

def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = np.array(std).reshape(3,1,1)
    mean = np.array(mean).reshape(3,1,1)
    return img * std + mean

def get_head_box_channel(x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False):
    head_box = np.array([x_min/width, y_min/height, x_max/width, y_max/height])*resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution-1)
    if coordconv:
        unit = np.array(range(0,resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit+i]) # [0-223], [1-224] .... [223-446]
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution,resolution), dtype=np.float32)
        head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel

def argmax_pts(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    print(heatmap.argmax(), heatmap.shape)
    print(idx)
    pred_y, pred_x = map(float,idx)
    return pred_x, pred_y

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return img

def multi_hot_targets(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int,[p[0]*w.float(), p[1]*h.float()])
            x = min(x, w-1)
            y = min(y, h-1)
            target_map[y, x] = 1
    return target_map

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
    ## groupby
    # csv_path = "/home/balin/exper/gaze/attention-target-detection/data/gazefollow/test_annotations_release.txt"
    # column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
    #                 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta']
    # df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
    # print(len(df))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)
    # df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
    #         'bbox_y_max']].groupby(['path', 'eye_x'])
    # print(len(df))

    # c = 0
    # for i in df:
    #     if c < 5:
    #         print(i)
    #         for j in i:
    #             print(j)
    #         c += 1

    # print(list(df.groups.keys()))

    ## head_box
    # unit = np.array(range(0,224), dtype=np.float32)
    # head_channel = []
    # for i in unit:
    #     head_channel.append([unit+i])
    # print(head_channel)

    ## gaussian
    # gaze_heatmap = np.zeros((1000, 1000))  # set the size of the output
    # gaze_heatmap = draw_labelmap(gaze_heatmap, [400, 800], 20, type='Gaussian')
    # # print(gaze_heatmap)
    # cv2.imwrite('color_img.jpg', gaze_heatmap)
    # cv2.imshow("image", gaze_heatmap)
    # cv2.waitKey()

    ## ROC
    # y_true = np.array([0, 0, 1, 1])
    # y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    # roc = roc_auc_score(y_true, y_scores)
    # print(roc)

    ## argmax
    # gaze_heatmap = np.zeros((48, 10, 10))
    # gaze_heatmap[0][4][5] = 1
    # x,y = argmax_pts(gaze_heatmap[0])

    ## mul
    # a = torch.randn(2, 1)
    # b = torch.randn(2, 2)
    # c = torch.mul(a, b)
    # print(c.size())
    # print(a)
    # print(b)
    # print(c)

    ## Involution
    stride = 1
    midchannel = 7
    inputchannel = 32
    batchsize = 8
    device = torch.device('cuda', 0)

    if torch.cuda.is_available():
        i = torch.randn(batchsize, inputchannel, 4, 4).cuda(device)
        conv2 = involution(inputchannel, midchannel, stride).cuda(device)
    else:
        i = torch.randn(batchsize, inputchannel, 4, 4)
        conv2 = involution(inputchannel, midchannel, stride)

    a = conv2(i)
    print(a)
    print(a.shape)