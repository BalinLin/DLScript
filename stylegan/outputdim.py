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
from training.networks_stylegan3 import *
import pickle

if __name__ == "__main__":
    # with open('stylegan3-t-ffhqu-256x256.pkl', 'rb') as f:
    #     G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    device = torch.device('cuda', 0)
    G = Generator(512,0,512,64,1).to(device)
    print(G.z_dim)
    print(G.c_dim)
    print(G.w_dim)
    print(G.img_resolution)
    print(G.img_channels)
    # print(G.mapping_kwargs)
    # print(G.synthesis_kwargs)
    z = torch.randn([1, G.z_dim]).cuda()    # latent codes
    c = None                                # class labels (not used in this example)
    img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1], no truncation
    print("="*50)
    print(torch.min(img), torch.max(img))
    print(img.shape)

    img = img.squeeze(0)
    img = torch.clamp(img, min=0, max=1)
    img = img.cpu()
    img = img.permute(1, 2, 0)

    plt.imshow(img.detach().numpy())
    plt.show()

    w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
    img = G.synthesis(w, noise_mode='const', force_fp32=True)
    print("="*50)
    print(torch.min(img), torch.max(img))
    print(img.shape)

    img = img.squeeze(0)
    img = torch.clamp(img, min=0, max=1)
    img = img.cpu()
    img = img.permute(1, 2, 0)
    plt.imshow(img.detach().numpy())
    plt.show()