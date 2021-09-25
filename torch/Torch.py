#! /usr/bin/env python3
# coding=utf-8

import cv2, torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
import torchvision.models as models


if __name__ == "__main__":
    a = torch.randn(3,4)
    print(a)
    b = torch.randn(3,1)
    print(b)
    c = torch.mul(a,b)
    print(c)

    print(a.requires_grad)
    print(b.requires_grad)
    print(c.requires_grad)