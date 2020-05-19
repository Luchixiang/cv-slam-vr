import torch
import torch.nn as nn
import numpy as np


def weight_decay(network, l2_value):  # 权重衰退
    decay, no_decay = [], []
    for param in network.parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]


def image2color(img):
    label_to_color = {
        0: [255, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 0],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30],
        7: [220, 220, 0],
        8: [107, 142, 35],
        9: [152, 251, 152],
        10: [70, 130, 180],
        11: [220, 20, 60],
        12: [255, 0, 0],
        13: [0, 0, 142],
        14: [0, 0, 70],
        15: [0, 60, 100],
        16: [0, 80, 100],
        17: [0, 0, 230],
        18: [119, 11, 32],
        19: [81, 0, 81]
    }
    img_height, img_width = img.shape
    print(img_height, img_width)
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for height in range(img_width):
            label = img[row][height]
            img_color[row][height] = label_to_color[label]

    return img_color
