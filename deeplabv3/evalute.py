import torch
import torch.nn as nn
import torch.utils.model_zoo
from utils.utils import image2color
import numpy as np
from model.deeplabv3 import DeepLabv3
import os
import cv2

batch_size = 1
print(os.getcwd())
network = DeepLabv3('eval', '/home/lcx/deeplabv3').cuda()
# network.load_state_dict(torch.load('/home/lcx/deeplabv3_pytorch/deeplabv3/pretrained_models/model_13_2_2_2_epoch_580.pth'))
state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth')
network.load_state_dict(state_dict)
img = cv2.imread('/home/lcx/deeplabv3/test2.JPG')
img = cv2.resize(img, (512, 1024), interpolation=cv2.INTER_NEAREST)
img = img / 255.0
img = img - np.array([0.485, 0.456, 0.406])
img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
img = img.astype(np.float32)
img = np.expand_dims(img, axis=0)
loss_fn = nn.CrossEntropyLoss()
network.eval()

with torch.no_grad():
    img = torch.from_numpy(img).cuda()
    img = img.float()
    # label_img =
    output = network(img)
    # loss = loss_fn()
    output = output.data.cpu().numpy()
    pred_label_img = np.argmax(output, axis=1)  # (shape: (batch_size, img_h, img_w))
    pred_label_img = pred_label_img.astype(np.uint8)
    img = np.squeeze(img, axis=0)
    pred_label_img = np.squeeze(pred_label_img, axis=0)
    img = img.data.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # (shape: (img_h, img_w, 3))
    img = img * np.array([0.229, 0.224, 0.225])
    img = img + np.array([0.485, 0.456, 0.406])
    img = img * 255.0
    img = img.astype(np.uint8)

    pred_label_img_color = image2color(pred_label_img)
    print(pred_label_img[200][200])

    overlayed_img = 0.35 * img + 0.65 * pred_label_img_color
    overlayed_img = overlayed_img.astype(np.uint8)

    cv2.imwrite(network.model_dir + "/" + 'test' + "_overlayed.png", pred_label_img_color)
