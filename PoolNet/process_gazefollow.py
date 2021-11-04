import torch
import torchvision.utils as vutils
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
from networks.poolnet import build_model, weights_init
from collections import OrderedDict
import scipy.misc as sm
import numpy as np
import os
import cv2
import math
import time
import urllib.request
import matplotlib.pyplot as plt

mode = "train"
channel = 1
show = False
save = True
dockerPath = True

# path
home = os.path.expanduser("~")
if mode == "test":
    load_dir = os.path.join(home, "exper/gaze/attention-target-detection/data/gazefollow/test2")
    save_dir = os.path.join(home, "exper/gaze/attention-target-detection/data/gazefollow/test2_sal")
    if dockerPath:
        load_dir = "/exper/gaze/attention-target-detection/data/gazefollow/test2"
        save_dir = "/exper/gaze/attention-target-detection/data/gazefollow/test2_sal"
elif mode == "train":
    load_dir = os.path.join(home, "exper/gaze/attention-target-detection/data/gazefollow/train")
    save_dir = os.path.join(home, "exper/gaze/attention-target-detection/data/gazefollow/train_sal")
    if dockerPath:
        load_dir = "/exper/gaze/attention-target-detection/data/gazefollow/train"
        save_dir = "/exper/gaze/attention-target-detection/data/gazefollow/train_sal"

os.mkdir(save_dir)

pretraind_model = "/exper/salient/PoolNet/dataset/run-0/run-0/models/final.pth" if dockerPath else os.path.join(home, "exper/salient/PoolNet/dataset/run-0/run-0/models/final.pth")
print('Loading pre-trained model from %s...' % pretraind_model)
net = build_model("resnet")
net.load_state_dict(torch.load(pretraind_model))

# gpu eval if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net.to(device)
net.eval()

for foldername in os.listdir(load_dir):
    print("foldername: ", foldername)
    load_foldername = os.path.join(load_dir, foldername)
    save_foldername = os.path.join(save_dir, foldername)
    os.mkdir(save_foldername)
    for filename in os.listdir(load_foldername):
        load_filename = os.path.join(load_foldername, filename)
        save_filename = os.path.join(save_foldername, filename)

        # load img
        img = cv2.imread(load_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        images = Variable(img).unsqueeze(0).type(torch.FloatTensor)
        if torch.cuda.is_available():
            images = images.to(device)

        # predict
        with torch.no_grad():
            preds = net(images)
            pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
            multi_fuse = 255 * pred

        if save:
            cv2.imwrite(save_filename, multi_fuse)
        if show:
            cv2.imshow(filename, multi_fuse)
            cv2.waitKey(0)
