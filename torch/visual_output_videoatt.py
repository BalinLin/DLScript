from posixpath import split
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from utils import imutils
from utils import myutils
from model import ModelSpatial
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from tqdm import tqdm


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((224, 224)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

transform = _get_transform()

home = os.path.expanduser("~")
home = "/"
load_dir = os.path.join(home, "exper/TestScript/attention-target-detection/images/test2")
save_dir = os.path.join(home, "exper/TestScript/attention-target-detection/images/test2_output")
save_dir_heatmap = os.path.join(home, "exper/TestScript/attention-target-detection/images/test2_heatmap")
csv_path =os.path.join(home, "exper/TestScript/attention-target-detection/images/test_annotations_release.txt")

os.mkdir(save_dir)
os.mkdir(save_dir_heatmap)
for foldername in os.listdir(load_dir):
    load_foldername = os.path.join(load_dir, foldername)
    save_foldername = os.path.join(save_dir, foldername)
    save_foldername_heatmap = os.path.join(save_dir_heatmap, foldername)
    os.mkdir(save_foldername)
    os.mkdir(save_foldername_heatmap)


def generate_data_field(eye_point):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 224, 224
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = eye_point
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    grid /= norm
    return grid

def preprocess_image(image_path, head):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]

    face_image = image[int(head[1]):int(head[3]), int(head[0]):int(head[2]), :]
    # process face_image for face net
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_image)
    face_image = transform(face_image)
    # process image for saliency net
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image)

    # generate head image
    head_channel = imutils.get_head_box_channel(head[0], head[1], head[2], head[3], w, h,
                                                    resolution=224, coordconv=False).unsqueeze(0)
    sample = {'image' : image,
              'face_image': face_image,
              'head': head_channel}

    return sample


def test(net, test_image_path, head):

    net.eval()
    heatmaps = []

    data = preprocess_image(test_image_path, head)

    image, face_image, head = data['image'], data['face_image'], data['head']
    image, face_image, head = map(lambda x: Variable(x.unsqueeze(0).cuda(), volatile=True), [image, face_image, head])

    predict_heatmap, attmap, inout_pred = net(image, head, face_image)

    final_output = predict_heatmap.cpu().data.numpy()

    heatmap = final_output.reshape([64, 64])

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 64., h_index / 64.])

    return heatmap, f_point[0], f_point[1]

def draw_result(image_path, eye, heatmap, gaze_point, head, gt):
    image_path = os.path.join(home, "exper/TestScript/attention-target-detection/images", image_path)
    pre_color = [179, 252, 17]
    gt_color = [0, 215, 255]
    x1, y1 = eye
    x2, y2 = gaze_point
    im = cv2.imread(image_path)
    image_height, image_width = im.shape[:2]
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    # cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 5, pre_color, -1)
    cv2.line(im, (x1, y1), (x2, y2), pre_color, 3)
    cv2.rectangle(im, (int(head[0]), int(head[1])), (int(head[2]), int(head[3])), pre_color, 3)

    gt_x = 0
    gt_y = 0
    c = 0

    for x, y in gt:
        if x != -1:
            gt_x += x
            gt_y += y
            c += 1
            # x, y = int(image_width * x), int(y * image_height)
            # cv2.circle(im, (x, y), 5, gt_color, -1)
            # cv2.line(im, (x1, y1), (x, y), gt_color, 3)

    gt_x /= c
    gt_y /= c
    gt_x = int(gt_x * image_width)
    gt_y = int(gt_y * image_height)

    cv2.circle(im, (gt_x, gt_y), 5, gt_color, -1)
    cv2.line(im, (x1, y1), (gt_x, gt_y), gt_color, 3)

    # heatmap visualization
    heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    heatmap = cv2.resize(heatmap, (image_width, image_height))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = (0.3 * heatmap.astype(np.float32) + 0.7 * im.astype(np.float32)).astype(np.uint8)
    img = np.concatenate((im, heatmap), axis=1)
    save_path = os.path.join(home, "exper/TestScript/attention-target-detection/images/test2_output", image_path.split('/')[-2], image_path.split('/')[-1])
    save_path_heatmap = os.path.join(home, "exper/TestScript/attention-target-detection/images/test2_heatmap", image_path.split('/')[-2], image_path.split('/')[-1])
    cv2.imwrite(save_path, im)
    cv2.imwrite(save_path_heatmap, heatmap)

    return img

def main():

    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load('model_demo.pt')
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()

    column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                    'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta']
    df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")

    df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
            'bbox_y_max']].groupby(['path', 'eye_x'])

    keys = list(df.groups.keys()) # ['path', 'eye_x'] pair key
    X_test = df
    length = len(keys)

    for index in range(length):
        g = X_test.get_group(keys[index]) # label of ['path', 'eye_x'] pair
        cont_gaze = []
        for i, row in g.iterrows():
            path = row['path']
            x_min = row['bbox_x_min']
            y_min = row['bbox_y_min']
            x_max = row['bbox_x_max']
            y_max = row['bbox_y_max']
            eye_x = row['eye_x']
            eye_y = row['eye_y']
            gaze_x = row['gaze_x']
            gaze_y = row['gaze_y']
            cont_gaze.append([gaze_x, gaze_y])  # all ground truth gaze are stacked up
        for j in range(len(cont_gaze), 20):
            cont_gaze.append([-1, -1])  # pad dummy gaze to match size for batch processing
        cont_gaze = torch.FloatTensor(cont_gaze)

        test_image_path = path
        x = eye_x
        y = eye_y
        headposition = (x_min, y_min, x_max, y_max)
        print(test_image_path, x, y)

        heatmap, p_x, p_y = test(model, os.path.join("./images", test_image_path), headposition)
        draw_result(test_image_path, (x, y), heatmap, (p_x, p_y), headposition, cont_gaze)

if __name__ == '__main__':
    main()

