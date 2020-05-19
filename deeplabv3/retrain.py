import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data

from dataset_own import DataTrain
from dataset_own import DatasetVal
from model.RDnet import RDnet
from utils.utils import weight_decay


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power):  # ploy_learning_rate in paper
    if iter > max_iter:
        return optimizer
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
model_id = '6'
num_epochs = 50
num_epochs_unfreeze = 100
batch_size = 3
learning_rate = 0.0001

network = RDnet(model_id=model_id, model_dir=os.getcwd()).to(device)
params = weight_decay(network, 0.0001)  # weight decay
optimizer = torch.optim.Adam(params, lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()

train_dataset = DataTrain(img_path='/home/lcx/deeplabv3/dataset/road')
val_dataset = DatasetVal(img_path='/home/lcx/deeplabv3/dataset/road')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=1)
epoch_losses_train = []
epoch_losses_val = []

# freeze
# for param in network.parameters():
#     param.requires_grad = False
# for param in network.con1x1.parameters():
#     param.requires_grad = True
#
# for epoch in range(num_epochs):
#     print('start new epoch')
#     print("epoch: %d/%d" % (epoch + 1, num_epochs))
#     network.train()
#     batch_losses = []
#     for step, (img, label_img) in enumerate(train_loader):
#         img = img.cuda()
#         label_img = label_img.long().cuda()
#         outputs = network(img)
#         loss = loss_fn(outputs, label_img)
#         loss_value = loss.data.cpu().numpy()
#         batch_losses.append(loss_value)
#         print("%d loss" % step, loss_value)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_train.append(epoch_loss)
#     with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_train, file)
#     print("train loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_train, "k^")
#     plt.plot(epoch_losses_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train loss per epoch")
#     plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
#     plt.close(1)
#
#     poly_lr_scheduler(optimizer, learning_rate, epoch, num_epochs, 0.99)
#
#     network.eval()
#     batch_losses = []
#     for step, (img, label_img) in enumerate(val_loader):
#         img = img.cuda()
#         label_img = label_img.long().cuda()
#         outputs = network(img)
#         loss = loss_fn(outputs, label_img)
#         loss_value = loss.data.cpu().numpy()
#         batch_losses.append(loss_value)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_val.append(epoch_loss)
#     with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_val, file)
#     print("val loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_val, "k^")
#     plt.plot(epoch_losses_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val loss per epoch")
#     plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
#     plt.close(1)
#
#     checkpoint_path = network.deeplabv3.checkpoints_dir + "/model_" + model_id + "_epoch_" + str(epoch + 1) + ".pth"
#     torch.save(network.state_dict(), checkpoint_path)

# unfreeze
network.load_state_dict(torch.load('/home/lcx/deeplabv3/training_logs/model_5/checkpoints/freeze_model.pth'))
for param in network.parameters():
    param.requires_grad = True

for epoch in range(50, num_epochs_unfreeze):
    print('start new epoch')
    print("epoch: %d/%d" % (epoch + 1, num_epochs_unfreeze))
    network.train()
    batch_losses = []
    for step, (img, label_img) in enumerate(train_loader):
        img = img.to(device)
        label_img = label_img.long().to(device)
        outputs = network(img)
        loss = loss_fn(outputs, label_img)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)
        print(" %d loss" % step, loss_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train_unfreeze.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train_unfreeze.png" % network.model_dir)
    plt.close(1)

    poly_lr_scheduler(optimizer, learning_rate, epoch, num_epochs, 0.99)

    network.eval()
    batch_losses = []
    for step, (img, label_img) in enumerate(val_loader):
        img = img.to(device)
        label_img = label_img.long().to(device)
        outputs = network(img)
        loss = loss_fn(outputs, label_img)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val_unfreeze.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print("val loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("%s/epoch_losses_val_unfreeze.png" % network.model_dir)
    plt.close(1)

    checkpoint_path = network.deeplabv3.checkpoints_dir + "/model_" + model_id + "_epoch_" + str(epoch + 1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)
