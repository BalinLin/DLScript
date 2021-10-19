#! /usr/bin/env python3
# coding=utf-8

import cv2, torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn as nn
# import yaml
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
import torchvision.models as models

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


if __name__ == "__main__":
    ## gaussian
    mse_loss = nn.MSELoss() # not reducing in order to ignore outside cases
    # mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases

    width, heigh = 180, 180
    eye_x, eye_y = 0.0, 0.0

    gaze_x, gaze_y = 0.2, 0.2
    direction_x, direction_y = gaze_x - eye_x + 1, gaze_y - eye_y + 1 # [-1 ~ 1] to [0 ~ 2]
    gaze_heatmap = np.zeros((width, heigh))  # set the size of the output
    gaze_heatmap = draw_labelmap(gaze_heatmap, [int(direction_x * width / 2), int(direction_y * heigh / 2)], 10, type='Gaussian')
    print(gaze_heatmap.shape)

    pre_gaze_x, pre_gaze_y = 0.7, 0.7
    pre_direction_x, pre_direction_y = pre_gaze_x + 1 - eye_x, pre_gaze_y - eye_y + 1 # [-1 ~ 1] to [0 ~ 2]
    pre_gaze_heatmap = np.zeros((width, heigh))  # set the size of the output
    pre_gaze_heatmap = draw_labelmap(pre_gaze_heatmap, [int(pre_direction_x * width / 2), int(pre_direction_y * heigh / 2)], 10, type='Gaussian')
    print(pre_gaze_heatmap.shape)

    loss = mse_loss(torch.tensor(pre_gaze_heatmap), torch.tensor(gaze_heatmap))
    print(loss.shape)
    print(loss)

    cv2.imwrite('gaze_heatmap.jpg', 255 * gaze_heatmap)
    cv2.imshow("image", gaze_heatmap)
    cv2.waitKey()

    cv2.imwrite('pre_gaze_heatmap.jpg', 255 * pre_gaze_heatmap)
    cv2.imshow("image", pre_gaze_heatmap)
    cv2.waitKey()