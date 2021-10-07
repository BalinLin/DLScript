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
    a = torch.randn(3,4)
    print(a)
    b = torch.randn(3,1)
    print(b)
    c = torch.mul(a,b)
    print(c)

    print(a.requires_grad)
    print(b.requires_grad)
    print(c.requires_grad)

    val_angle_loss = torch.tensor(float('inf'))
    print("val_angle_loss: ", val_angle_loss)

    val_angle_loss_temp = torch.tensor(1)
    print(val_angle_loss < val_angle_loss_temp)
    print(val_angle_loss > val_angle_loss_temp)
    print(torch.gt(val_angle_loss, val_angle_loss_temp))
    print(torch.gt(val_angle_loss_temp, val_angle_loss))


    val_angle_loss = torch.min(val_angle_loss, val_angle_loss_temp)
    print("val_angle_loss: ", val_angle_loss)

