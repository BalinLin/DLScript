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
    print("input")
    print(input)
    output = torch.mean(input)
    print("output")
    print(output)
    output = torch.mean(input, dim=0)
    print("output")
    print(output)