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
    a = torch.randn(2,3,4)
    print(a)
    print(a.shape)
    print(a[:,:,:])
    print(a[:,:,:].shape)
    print(a[0,:,:])
    print(a[0,:,:].shape)
    print(a[0,0,:])
    print(a[0,0,:].shape)
    print(a[0,0,2])
    print(a[0,0,2].shape)
