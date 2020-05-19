import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data

from dataset import DataTrain
from dataset import DatasetVal
from model.deeplabv3 import DeepLabv3
from utils.utils import weight_decay


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power):  # ploy_learning_rate in paper
    if iter > max_iter:
        return optimizer
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


model_id = '4'
num_epochs = 1000
batch_size = 3
learning_rate = 0.0001

network = DeepLabv3(model_id=model_id, project_dir=os.getcwd()).cuda()
params = weight_decay(network, 0.0001)  # weight decay
optimizer = torch.optim.Adam(params, lr=learning_rate)

with open("./dataset/cityscape/meta/class_weights.pkl", "rb") as file:
    class_weights = np.array(pickle.load(file))
class_weights = torch.from_numpy(class_weights)  # class weight
class_weights = class_weights.float().cuda()
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

train_dataset = DataTrain(cityscapes_data_path='./dataset/cityscape',
                          cityscapes_meta_path='./dataset/cityscape/meta')
val_dataset = DatasetVal(cityscapes_data_path='./dataset/cityscape',
                         cityscapes_meta_path='./dataset/cityscape/meta')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=1)
epoch_losses_train = []
epoch_losses_val = []
for epoch in range(num_epochs):
    print('start new epoch')
    print("epoch: %d/%d" % (epoch + 1, num_epochs))
    network.train()
    batch_losses = []
    for step, (img, label_img) in enumerate(train_loader):
        img = img.cuda()
        label_img = label_img.long().cuda()
        outputs = network(img)
        loss = loss_fn(outputs, label_img)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)
        print("第%d步loss" % step, loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
    plt.close(1)

    poly_lr_scheduler(optimizer, learning_rate, epoch, num_epochs, 0.99)

    network.eval()
    batch_losses = []
    for step, (img, label_img,img_id) in enumerate(val_loader):
        img = img.cuda()
        label_img = label_img.long().cuda()
        outputs = network(img)
        loss = loss_fn(outputs, label_img)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print("val loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
    plt.close(1)

    checkpoint_path = network.checkpoints_dir + "/model_" + model_id + "_epoch_" + str(epoch + 1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)
