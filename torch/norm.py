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
    a = torch.randn(10,3)
    print(a)
    b = torch.randn(10,3)
    print(b)
    c = a + b
    print(c)
    norm = torch.norm(c[:, :2], 2, dim=1)
    print(norm)