#! /usr/bin/env python3
# coding=utf-8

import cv2, torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter
import torchvision.models as models


if __name__ == "__main__":
    batch = 10
    input_size = 224
    img = torch.randn(batch,3,224,224)
    print("img")
    label = torch.randn(batch,20,2)
    for b_i in range(len(label)):
        valid_gaze = label[b_i]
        for gt_gaze in valid_gaze:
            x, y = int(gt_gaze[0] * input_size), int(gt_gaze[1] * input_size)
            x = max(x, 0)
            x = min(x, 223)
            y = max(y, 0)
            y = min(y, 223)
            print(img[b_i, 0, x, y])
