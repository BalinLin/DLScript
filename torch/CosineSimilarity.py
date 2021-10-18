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
    cosine_similarity = torch.nn.CosineSimilarity()
    input = torch.randn(3, 2, requires_grad=True)
    target = torch.randn(3, 2)
    input = torch.tensor([[10., -10.], [1., -1.]])
    target = torch.tensor([[1., 1.], [-1., 1.]])
    angle_loss = torch.mean(1 - cosine_similarity(input, target))
    print(input)
    print(target)
    print("cosine_similarity(input, target)")
    print(cosine_similarity(input, target))
    print("angle_loss")
    print(angle_loss)