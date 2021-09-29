#! /usr/bin/env python3
# coding=utf-8

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter

if __name__ == "__main__":
    path = "train/00000080/00080697.jpg"
    print(path.split("/"))
    home = os.path.expanduser("~")
    o = os.path.join(home, path.split("/")[1], path.split("/")[2])
    print(o)