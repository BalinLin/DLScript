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
    L1_loss = torch.nn.L1Loss(reduction='mean')
    L2_loss = torch.nn.MSELoss()
    input = torch.randn(3, 1, requires_grad=True)
    target = torch.randn(3, 1)
    output = L1_loss(input, target)
    print(input)
    print(target)
    print("L1")
    print(output)
    output.backward()

    output = L2_loss(input, target)
    print("L2")
    print(output)
    output.backward()