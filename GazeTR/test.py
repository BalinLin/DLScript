from model import Model
import torch
from torchvision import models
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import os

input_resolution = 224
output_resolution = 224
gazefollow_train_data = "/exper/gaze/attention-target-detection/data/gazefollow"
gazefollow_train_label = "/exper/gaze/attention-target-detection/data/gazefollow/train_annotations_release.txt"
save_dir = "/exper/TestScript/GazeTR/visual/"

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

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    # transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


class GazeFollow(Dataset):
    def __init__(self, data_dir, csv_path, transform, input_size=input_resolution, output_size=output_resolution,
                 test=False, imshow=False):

        column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                        'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta']
        df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        df = df[df['inout'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
        df.reset_index(inplace=True)
        self.y_train = df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'eye_x', 'eye_y', 'gaze_x',
                            'gaze_y', 'inout']]
        self.X_train = df['path']
        self.length = len(df)

        self.data_dir = data_dir
        self.transform = transform
        self.test = test

        self.input_size = input_size
        self.output_size = output_size
        self.imshow = imshow

    def __getitem__(self, index):

        path = self.X_train.iloc[index]
        x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, inout = self.y_train.iloc[index]
        gaze_inside = bool(inout)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max]) # map type to float

        if self.imshow:
            img.save("origin_img.jpg")

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        if self.imshow:
            img.save("img_aug.jpg")
            face.save('face_aug.jpg')

        if self.transform is not None:
            img = self.transform(img)
            face = self.transform(face)

        return img, face

    def __len__(self):
        return self.length

transform = _get_transform()

# Prepare data
print("Loading Data")
train_dataset = GazeFollow(gazefollow_train_data, gazefollow_train_label,
                    transform, input_size=input_resolution, output_size=output_resolution)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=32,
                                            shuffle=True,
                                            num_workers=0)


device = torch.device('cuda', 0)
GazeTR = Model().to(device)
# print(GazeTR)
print("memory:", torch.cuda.max_memory_allocated())

model_dict = GazeTR.state_dict()
pretrained_dict = torch.load("./GazeTR-H-ETH.pt")
model_dict.update(pretrained_dict)
GazeTR.load_state_dict(model_dict)

# for key in model_dict:
    # print(key)
    # layer_name, weights = new[count]
    # model_dict[key] = weights
    # count += 1
    # model.load_state_dict(model_dict)

# img = torch.ones(10, 3, 224 ,224).cuda().to(device)

for batch, (image, face) in enumerate(train_loader):
    # load image
    image = image.cuda().to(device)
    face = face.cuda().to(device)
    # img.save("origin_img.jpg")
    input = {'face': image}
    input = {'face': face}

    # for test
    gaze = GazeTR(input)
    # print(gaze)
    # print(gaze.shape)
    # print(torch.max(gaze), torch.min(gaze))

    # utils.save_image(gaze, "gaze.jpg")
    utils.save_image(image, "test.jpg")

    # show gaze direction with image
    for i in range(len(gaze)):
        fig, ax = plt.subplots(1, 3)

        eye = [0.5,0.5]
        gaze_field = generate_data_field(eye)
        gaze_field = torch.tensor(gaze_field)
        x_pos = 0
        y_pos = 0
        x_direct = gaze[i][0].cpu().detach().numpy()
        y_direct = gaze[i][1].cpu().detach().numpy()
        print("x_direct:", x_direct)
        print("y_direct:", y_direct)

        direction = [[float(-x_direct), float(-y_direct)]]
        direction = torch.tensor(direction)
        norm = torch.norm(direction, 2, dim=1)
        normalized_direction = direction / norm.view([-1, 1])
        normalized_direction = torch.tensor(normalized_direction)

        # generate gaze field map
        channel, height, width = gaze_field.size()
        gaze_field = gaze_field.permute([1, 2, 0]).contiguous()
        gaze_field = gaze_field.view([-1, 2])
        gaze_field = torch.matmul(gaze_field, normalized_direction.view([2, 1]))
        gaze_field_map = gaze_field.view([height, width, 1])
        gaze_field_map = gaze_field_map.permute([2, 0, 1]).contiguous()
        gaze_field_map = torch.clip(gaze_field_map, 0, 1)

        img = np.transpose(face[i].cpu().detach().numpy(), (1, 2, 0))
        ax[0].imshow(img)
        ax[1].quiver(x_pos, y_pos, x_direct, y_direct)
        ax[1].set_title('Quiver plot with one arrow')
        ax[2].imshow(transforms.ToPILImage()(gaze_field_map).convert('RGB'))

        plt.show()