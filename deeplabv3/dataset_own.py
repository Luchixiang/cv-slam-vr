import torch
import torch.utils.data

import numpy as np
import cv2
import os
import glob


class DataTrain(torch.utils.data.Dataset):
    def __init__(self, img_path):
        self.img_dir = img_path + '/train'
        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024
        self.examples = []

        train_img_dir = self.img_dir
        for dir_name in glob.glob(train_img_dir + '/*_json'):
            example = {}
            img_path = dir_name + '/img.png'
            label_img_path = dir_name + '/label_img.png'
            example['img_path'] = img_path
            example['label_img_path'] = label_img_path
            self.examples.append(example)
        self.num_examples = len(self.examples)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        example = self.examples[index]
        img_path = example['img_path']
        label_img_path = example['label_img_path']
        img = cv2.imread(img_path, -1)  # (1024,2048,3)
        label_img = cv2.imread(label_img_path, -1)  # (1024,2048)
        # resize the img
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST)
        label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                               interpolation=cv2.INTER_NEAREST)
        # data augment
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)
            label_img = cv2.flip(label_img, 1)

        ########################################################################
        # randomly scale the img and the label:
        ########################################################################
        # scale = np.random.uniform(low=0.7, high=2.0)
        # new_img_h = int(scale * self.new_img_h)
        # new_img_w = int(scale * self.new_img_w)
        #
        # img = cv2.resize(img, (new_img_w, new_img_h),
        #                  interpolation=cv2.INTER_NEAREST)  # (shape: (new_img_h, new_img_w, 3))
        #
        # label_img = cv2.resize(label_img, (new_img_w, new_img_h),
        #                        interpolation=cv2.INTER_NEAREST)  # (shape: (new_img_h, new_img_w))
        # 归一化
        img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)

        img = torch.from_numpy(img)
        label_img = torch.from_numpy(label_img)
        return img, label_img


class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, img_path):
        self.img_dir = img_path + '/train'

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []

        for dir_name in glob.glob(self.img_dir + '/*_json'):
            example = {}
            img_path = dir_name + '/img.png'
            label_img_path = dir_name + '/label_img.png'
            example['img_path'] = img_path
            example['label_img_path'] = label_img_path
            self.examples.append(example)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]


        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)  # (shape: (1024, 2048, 3))
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1)  # (shape: (1024, 2048))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                               interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024))

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)  # (shape: (3, 512, 1024))
        label_img = torch.from_numpy(label_img)  # (shape: (512, 1024))

        return img, label_img

    def __len__(self):
        return self.num_examples
