import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
# from gazenet import GazeNet

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging

from scipy import signal

from utils import data_transforms
from utils import get_paste_kernel, kernel_map

def generate_data_field(eye_point):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 224, 224
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    print("x_grid:", x_grid.shape)
    print("x_grid:", x_grid)
    print("y_grid:", y_grid.shape)
    print("y_grid:", y_grid)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)
    print("grid:", grid.shape)
    print("grid:", grid)

    x, y = eye_point
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    print("grid:", grid.shape)
    print("grid:", grid)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    print("norm:", norm.shape)
    print("norm:", norm)
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    print("norm:", norm.shape)
    print("norm:", norm)
    grid /= norm
    print("grid:", grid.shape)
    print("grid:", grid)
    return grid

i = [0.4,0.6]
gaze_field = generate_data_field(i)
gaze_field = torch.tensor(gaze_field)
direction = [[-0.1, 0.05]]
direction = torch.tensor(direction)
norm = torch.norm(direction, 2, dim=1)
normalized_direction = direction / norm.view([-1, 1])
normalized_direction = torch.tensor(normalized_direction)

# generate gaze field map
channel, height, width = gaze_field.size()
gaze_field = gaze_field.permute([1, 2, 0]).contiguous()
gaze_field = gaze_field.view([-1, 2])
print("gaze_field:", gaze_field.shape)
gaze_field = torch.matmul(gaze_field, normalized_direction.view([2, 1]))
print("gaze_field:", gaze_field.shape)
gaze_field_map = gaze_field.view([height, width, 1])
gaze_field_map = gaze_field_map.permute([2, 0, 1]).contiguous()
gaze_field_map = torch.clip(gaze_field_map, 0, 1)
print("gaze_field_map:", gaze_field_map.shape)
print("gaze_field_map:", gaze_field_map)

gaze_field_map_2 = torch.pow(gaze_field_map, 3)
print("gaze_field_map_2:", gaze_field_map_2.shape)
print("gaze_field_map_2:", gaze_field_map_2)

gaze_field_map_3 = torch.pow(gaze_field_map, 5)
print("gaze_field_map_3:", gaze_field_map_3.shape)
print("gaze_field_map_3:", gaze_field_map_3)

img = transforms.ToPILImage()(gaze_field_map).convert('RGB')
img.save("./img1.jpg")
img = transforms.ToPILImage()(gaze_field_map_2).convert('RGB')
img.save("./img2.jpg")
img = transforms.ToPILImage()(gaze_field_map_3).convert('RGB')
img.save("./img3.jpg")