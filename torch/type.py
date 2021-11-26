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
    input = torch.randn(3, 2, requires_grad=True)
    target = torch.randn(3, 2)
    output = L1_loss(input, target)
    print(input)
    print(target)
    print("L1")
    print(output)

    output = output + torch.FloatTensor([2])
    print(output)
    output = output.squeeze()
    print(output)
