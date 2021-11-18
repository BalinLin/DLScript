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
    input = torch.randn(3, 2)
    print("input:")
    print(input)
    idx = input > 1
    print("idx:")
    print(idx)
    input[idx] = 100
    print("input:")
    print(input)

    input2 = torch.randn(2, 1, 7, 7)
    print("input2:")
    print(input2)
    input2 = torch.clamp(input2, min=0, max=1)
    print("input2:")
    print(input2)
    idx = input2 > 0
    input2[idx] = 1
    print("input2:")
    print(input2)