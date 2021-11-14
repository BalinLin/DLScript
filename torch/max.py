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
    input = torch.randn(3, 2, requires_grad=True)
    target = torch.randn(3, 2)
    output = torch.max(0.1*input, -0.9*input)
    print(input)
    print(0.1*input)
    print(-0.9*input)
    print(output)