import numpy as np
import cv2
import os

path = '/home/lcx/deeplabv3/dataset/road/valid/'
paths = os.listdir(path)
num = 0
temp = []
for aa in paths:
    img = cv2.imread(path + aa + '/label_gray.png', -1)
    if img is not None:
        new_img = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == 0:
                    new_img[i][j] = 0
                if img[i][j] == 76:
                    new_img[i][j] = 1
                if img[i][j] == 29:
                    new_img[i][j] = 2
        cv2.imwrite(path + aa + '/label_img.png', new_img)
