import torch.nn as nn
import torch
import torch.nn.functional as F
from .deeplabv3 import DeepLabv3


class RDnet(nn.Module):
    def __init__(self, model_dir, model_id):
        super(RDnet, self).__init__()
        self.model_dir = model_dir
        self.model_id = model_id
        self.deeplabv3 = DeepLabv3(model_id=self.model_id, project_dir=self.model_dir)
        # self.deeplabv3.load_state_dict(torch.load('/home/lcx/.torch/models/deeplabv3_resnet101_coco-586e9e4e.pth'))
        self.deeplabv3.load_state_dict(torch.load('/home/lcx/deeplabv3/training_logs/model_3/checkpoints/model_3_epoch_8.pth'))
        self.con1x1 = nn.Conv2d(20, 3, kernel_size=1)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        x = self.deeplabv3(x)
        x = self.con1x1(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear')
        return x

