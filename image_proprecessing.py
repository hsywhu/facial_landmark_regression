from __future__ import print_function, division
import os
import torch
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings
import copy

from torchvision import transforms,utils

warnings.filterwarnings("ignore")


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

class LandmarksDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform= None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        #return len(self.root_dir)
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        'Ext Images'
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name) #maybe need / 255
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float')
        landmarks_2d = landmarks.reshape(-1, 2)
        sample = {'image': image, 'landmarks_2d': landmarks_2d}

        if self.transform:
            sample = self.transform(sample)
        return sample

# Transforms
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks_2d = sample['image'], sample['landmarks_2d']
        #print(image.shape)

        h, w = image.shape[:2]
        '''
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        '''
        new_h, new_w = int(self.output_size), int(self.output_size)

        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks_2d.reshape(-1,1)
        landmarks_2d = landmarks_2d* [new_w/ w, new_h/ h]
        return {'image': img, 'landmarks_2d': landmarks_2d}

class Normailize(object):
    def __call__(self, sample):
        image, landmarks_2d = sample['image'], sample['landmarks_2d']

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean)/ std
        return {'image': image, 'landmarks_2d':landmarks_2d}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, landmarks_2d = sample['image'], sample['landmarks_2d']
        h, w = image.shape[:2]
        #radn_num = random.uniform(0, 10)
        radn_num = 10
        if radn_num > 5:
            for i in range(image.shape[2]):
                image[:,:,i] = np.flip(image[:,:,i], 1)
            landmarks_2d[:, 0] = w - landmarks_2d[:, 0]
            new_landmarks_2d = copy.deepcopy(landmarks_2d)
            new_landmarks_2d[0, :] = landmarks_2d[3, :]
            new_landmarks_2d[1, :] = landmarks_2d[2, :]
            new_landmarks_2d[2, :] = landmarks_2d[1, :]
            new_landmarks_2d[3, :] = landmarks_2d[0, :]
            new_landmarks_2d[4, :] = landmarks_2d[5, :]
            new_landmarks_2d[5, :] = landmarks_2d[4, :]
            #print("landmark_2d size: ", landmarks_2d.shape)
        #print(image)
        return {'image':image, 'landmarks_2d':new_landmarks_2d}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks_2d = sample['image'], sample['landmarks_2d']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks_2d = landmarks_2d - [left, top]
        return {'image': image, 'landmarks_2d': landmarks_2d}

class ToTensor(object):
    def __call__(self, sample):
        image, landmarks_2d = sample['image'], sample['landmarks_2d']
        image = image.transpose((2, 0, 1))
        return{'image': torch.from_numpy(image),'landmarks_2d': torch.from_numpy(landmarks_2d)}
