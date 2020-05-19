import glob
import os
import time

import cv2
import numpy as np
import torch.utils.model_zoo

from model.RDnet import RDnet
from utils.utils import image2color

batch_size = 1
tt_time = 0
path = '/home/lcx/deeplabv3/testImg/img'
network = RDnet('eval', '/home/lcx/deeplabv3').cuda()
network.load_state_dict(torch.load('/home/lcx/deeplabv3/training_logs/model_5/checkpoints/model_5_epoch_99.pth'))
imgs = os.listdir(path)
num = 0
import glob
import os
import time

import cv2
import numpy as np
import torch.utils.model_zoo

from model.RDnet import RDnet
from utils.utils import image2color

batch_size = 1
tt_time = 0
network = RDnet('eval', '/home/zzk/deeplabv3').cuda()
network.load_state_dict(torch.load('./model_5_epoch_99.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network.to(device)
num = 0
cap = cv2.VideoCapture('./test2.mov')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.mov', fourcc, 40.0, (512, 1024))
while cap.isOpened():
    ret, img = cap.read()
    if ret is False:
        break
    if img is not None:
        start_time = time.time()
        img = cv2.resize(img, (512, 1024), interpolation=cv2.INTER_NEAREST)
        img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        network.eval()
        with torch.no_grad():
            img = torch.from_numpy(img).cuda()
            img = img.float().cuda()
            # label_img =
            output = network(img)
            # loss = loss_fn()
            output = output.data.cpu().numpy()
            pred_label_img = np.argmax(output, axis=1)  # (shape: (batch_size, img_h, img_w))
            pred_label_img = pred_label_img.astype(np.uint8)
            pred_label_img = np.squeeze(pred_label_img)
            img = np.squeeze(img)
            img = img.data.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))  # (shape: (img_h, img_w, 3))
            img = img * np.array([0.229, 0.224, 0.225])
            img = img + np.array([0.485, 0.456, 0.406])
            img = img * 255.0
            img = img.astype(np.uint8)
            num = num + 1
            pred_label_img_color = image2color(pred_label_img)
            overlayed_img = 0.35 * img + 0.65 * pred_label_img_color
            overlayed_img = overlayed_img.astype(np.uint8)
            end_time = time.time()
            print(end_time - start_time)
            # cv2.imwrite(network.model_dir + "/" + 'test%d' % num + "_overlayed.png", overlayed_img)
            out.write(overlayed_img)
cap.release()
out.release()