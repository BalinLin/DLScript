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
    loss = torch.nn.L1Loss(reduction='mean')
    input = torch.randn(3, 1, requires_grad=True)
    target = torch.randn(3, 1)
    output = loss(input, target)
    print(input)
    print(target)
    print(output)
    output.backward()