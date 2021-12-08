#! /usr/bin/env python3
# coding=utf-8

import cv2, torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter
import torchvision.models as models
from coord import CoordAtt

if __name__ == "__main__":
    input = torch.randn(2, 512, 7, 7, requires_grad=True)
    model = CoordAtt(inp=512, oup=512, groups=32)
    output = model(input)
    print(output)
    print(output.shape)