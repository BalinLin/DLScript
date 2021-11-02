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


if __name__ == "__main__":
    a = torch.randn(10,3)
    print(a)
    b = torch.randn(10,3)
    print(b)
    c = torch.stack((a, b))
    print(c.shape)

    listAll = []
    l = []
    l.append(a)
    l.append(b)
    d = torch.stack(l)
    print(d.shape)
    listAll.append(d)

    lengths = [i for i in range(10)]
    listAll.append(lengths)
    print(len(listAll))