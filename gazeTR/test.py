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
load_dir = "/exper/GazeTR/data/"
save_dir = "/exper/GazeTR/visual/"

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    # transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

class GazeFollow(Dataset):
    def __init__(self, data_dir, transform, input_size=input_resolution, output_size=output_resolution, imshow=False):

        self.data_dir = data_dir
        self.transform = transform
        self.input_size = input_size
        self.output_size = output_size
        self.imshow = imshow
        self.path = []
        for path in os.listdir(load_dir):
            self.path.append(path)
        self.length = len(self.path)


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.path[index]))
        img = img.convert('RGB')

        if self.imshow:
            img.save("origin_img.jpg")

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.length

transform = _get_transform()

# Prepare data
print("Loading Data")
train_dataset = GazeFollow(load_dir,
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

for batch, image in enumerate(train_loader):
    # load image
    image = image.cuda().to(device)
    # img.save("origin_img.jpg")
    img = {'face': image}

    # for test
    gaze = GazeTR(img)
    # print(gaze)
    # print(gaze.shape)
    # print(torch.max(gaze), torch.min(gaze))

    # utils.save_image(gaze, "gaze.jpg")
    utils.save_image(image, "test.jpg")

    # show gaze direction with image
    fig, ax = plt.subplots(1, 2)

    x_pos = 0
    y_pos = 0
    x_direct = gaze[0][0].cpu().detach().numpy()
    y_direct = gaze[0][1].cpu().detach().numpy()

    image = np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0))
    ax[0].imshow(image)
    ax[1].quiver(x_pos, y_pos, x_direct, y_direct)
    ax[1].set_title('Quiver plot with one arrow')


    plt.show()