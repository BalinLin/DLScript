#! /usr/bin/env python3
# coding=utf-8

from posix import listdir
import cv2, torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter
import torchvision.models as models
import math

if __name__ == "__main__":
    a = torch.randn(10,1)
    print(a)
    var = torch.nn.Sigmoid()(a)
    print(var)
    print(math.pi*var)
    print(var.shape)
    var = var.view(-1,1).expand(var.size(0),2)
    print(var.shape)
    print(var)