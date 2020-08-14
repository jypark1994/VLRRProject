#!/usr/bin/env python
# coding: utf-8

# In[1]:

# 1. dataset 의 transforms callback 을 통해 주어진 몇 개의 down_scale로 무작위 LR 이미지를 샘플링 할 수 있지 않을까?
# 2. 무작위 Multiscale 을 커버하기 위해 GT 이미지 크기로 LR 이미지를 다시 Interpolate 하지만 ... 다른 방법은 없을까?
# -- 모델 내부에서 이를 재해석 한다면?


import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim, utils
from torchvision import models, datasets, transforms

import lrResnet

import PIL.Image as Image

import random
import cv2
import os
import argparse

import time

parser = argparse.ArgumentParser()
parser.add_argument("--down_scale", type=int, default=1)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--model_name", type=str, default="resnet50")
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

print(args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device ...", device)

init_scale = 1.15

down_scale = args.down_scale

transforms_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1),
    transforms.RandomAffine(360,scale=[init_scale-0.15,init_scale+0.15]),
    transforms.CenterCrop(224),
    transforms.Resize(224//down_scale),
#     transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
])

transforms_test = transforms.Compose([
    transforms.Resize(224//down_scale),
#     transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.ToTensor()
])
    
train_dataset = datasets.ImageFolder("/media/data/MosquitoDL/TrainVal", transform=transforms_train)
train_loader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

test_dataset = datasets.ImageFolder("/media/data/MosquitoDL/Test", transform=transforms_test)
test_loader = utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

def get_model(model, pretrained=False, num_classes = 6):
    if model == 'resnet50':
        if down_scale == 1:
            net = models.resnet50(pretrained=pretrained)
        else:
            net = lrResnet.resnet50_LR(scale=down_scale, pretrained=pretrained)
    elif model == 'resnet34':
        if down_scale == 1:
            net = models.resnet34(pretrained=pretrained)
        else:
            net = lrResnet.resnet34_LR(scale=down_scale, pretrained=pretrained)
        net.fc = nn.Linear(in_features=512, out_features=num_classes)
        
    elif model == 'resnet18':
        if down_scale == 1:
            net = models.resnet18(pretrained=pretrained)
        else:
            net = lrResnet.resnet18_LR(scale=down_scale, pretrained=pretrained)
        net.fc = nn.Linear(in_features=512, out_features=num_classes)
    return net

model_name = args.model_name
net_t = get_model(model_name, pretrained=args.pretrained, num_classes=18)
net_t = net_t.to(device)

def train(net, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=1e-4, weight_decay=4e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    
    train_avg_loss = 0
    n_count = 0
    n_corrects = 0

    net.train()

    for j, data in enumerate(train_loader):
        batch, label = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        pred = net(batch)

        loss = criterion(pred, label)
        train_avg_loss += loss

        n_corrects += torch.sum(torch.argmax(pred, dim=1) == label).item()
        n_count += label.shape[0]

        loss.backward()
        optimizer.step()

    train_accuracy = n_corrects/n_count
    train_avg_loss /= n_count

    return train_accuracy, train_avg_loss, net

def evaluate(net ,test_loader):
    net.eval()
    
    n_count = 0
    n_corrects = 0

    for j, data in enumerate(test_loader):


        batch, label = data[0].to(device), data[1].to(device)

        pred = net(batch)

        n_corrects += torch.sum(torch.argmax(pred, dim=1) == label).item()
        n_count += label.shape[0]

    test_accuracy = n_corrects/n_count
    
    return test_accuracy, net

def train_and_eval(net, epochs, train_loader, test_loader, save_name='default.pth'):
    print("─── Start Training & Evalutation ───")
    
    best_accuracy = 0
    best_model = None
    
    for i in range(epochs):
        time_start = time.time()
        
        print(f"┌── Epoch ({i}/{epochs-1})")
        
        train_acc, loss, net = train(net, train_loader)
        print(f"├── Training Loss : {loss:.4f}")
        print(f'├── Training accuracy : {train_acc*100:.2f}%')
        print("│")
        test_acc, net = evaluate(net, test_loader)
        print(f'└── Testing accuracy : {test_acc*100:.2f}%')
        
        if test_acc > best_accuracy:
            print(f"  └──> Saving the best model to \"{save_name}\"")
            best_accuracy = test_acc
            best_model = net.state_dict()
            model_dict = {'acc':best_accuracy, 'net':best_model}
            torch.save(model_dict, save_name)
            
        time_end = time.time()
        
        epoch_time = time_end - time_start
        epoch_time_gm = time.gmtime(epoch_time)
        estimated_time = epoch_time * (epochs - 1 - i)
        estimated_time_gm = time.gmtime(estimated_time)
        print(f"Epoch time ─ {epoch_time_gm.tm_hour}[h] {epoch_time_gm.tm_min}[m] {epoch_time_gm.tm_sec}[s]")
        print(f"Estimated time ─ {estimated_time_gm.tm_hour}[h] {estimated_time_gm.tm_min}[m] {estimated_time_gm.tm_sec}[s]")
        print("\n")

    return best_accuracy, best_model
        
epochs = args.epochs

accuracy, net_t = train_and_eval(net_t, epochs, train_loader, test_loader, save_name=f"./models/mosquitoes/best_{model_name}_x{down_scale}_{args.pretrained}.pth")

print(f"Done with accuracy : {accuracy*100:.2f}%")




