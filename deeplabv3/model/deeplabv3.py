import os

import torch.nn as nn
import torch.nn.functional as F

from model.aspp import ASPP_Bottleneck
from model.resnet import ResNet101_OS16


class DeepLabv3(nn.Module):
    def __init__(self, model_id, project_dir):
        super(DeepLabv3, self).__init__()
        self.model_id = model_id
        self.project_dir = project_dir
        self.num_classes = 20
        self.create_model_dirs()
        self.resnet = ResNet101_OS16()
        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

    def forward(self, x):
        # x :(batch_size, 3, h, w)
        h = x.size()[2]
        w = x.size()[3]
        feature_map = self.resnet(x)
        output = self.aspp(feature_map)
        # output = F.interpolate(output, size=(h, w), mode="bilinear")  # 恢复图像分辨率

        return output

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        print(self.logs_dir)
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
