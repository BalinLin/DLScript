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

show = False
save = True
dockerPath = True
minHW = 112

# path
home = os.path.expanduser("~")
load_dir = os.path.join(home, "exper/gaze/attention-target-detection/data/videoatttarget/images")
save_dir = os.path.join(home, "exper/gaze/attention-target-detection/data/videoatttarget/images_sal")
if dockerPath:
    load_dir = "/exper/gaze/attention-target-detection/data/videoatttarget/images"
    save_dir = "/exper/gaze/attention-target-detection/data/videoatttarget/images_sal"


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

pretraind_model = "/exper/salient/PoolNet/dataset/run-0/run-0/models/final.pth" if dockerPath else os.path.join(home, "exper/salient/PoolNet/dataset/run-0/run-0/models/final.pth")
print('Loading pre-trained model from %s...' % pretraind_model)
net = build_model("resnet")
net.load_state_dict(torch.load(pretraind_model))

# gpu eval if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net.to(device)
net.eval()


for foldername_first in os.listdir(load_dir):
    load_foldername_first = os.path.join(load_dir, foldername_first)
    save_foldername_first = os.path.join(save_dir, foldername_first)
    if not os.path.exists(save_foldername_first):
        print("foldername_first: ", foldername_first)
        os.mkdir(save_foldername_first)
        for foldername_second in os.listdir(load_foldername_first):
            load_foldername_second = os.path.join(load_foldername_first, foldername_second)
            save_foldername_second = os.path.join(save_foldername_first, foldername_second)
            if not os.path.exists(save_foldername_second):
                print("    foldername_second: ", foldername_second)
                os.mkdir(save_foldername_second)

                for filename in os.listdir(load_foldername_second):
                    load_filename = os.path.join(load_foldername_second, filename)
                    save_filename = os.path.join(save_foldername_second, filename)

                    # load img
                    img = cv2.imread(load_filename)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height, width, channels = img.shape
                    m = min(height, width)
                    if m < minHW:
                        if height < width:
                            scale = minHW / height
                            img = cv2.resize(img, (int(width * scale), minHW), interpolation=cv2.INTER_CUBIC)
                        else:
                            scale = minHW / width
                            img = cv2.resize(img, (minHW, int(height * scale)), interpolation=cv2.INTER_CUBIC)

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
                        if m < minHW:
                            multi_fuse = cv2.resize(multi_fuse, (width, height), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(save_filename, multi_fuse)
                    if show:
                        cv2.imshow(filename, multi_fuse)
                        cv2.waitKey(0)
            else:
                print("    foldername_second %s exist.", foldername_second)
    else:
        print("foldername_first %s exist.", foldername_first)
