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
from torchvision import models, transforms

from model_cifar.resnet_cifar import resnet32
import datasets

import PIL.Image as Image

import random
import cv2
import os
import argparse

import time

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--model_name", type=str, default="resnet18")
parser.add_argument("--down_scale", type=int, default=1)
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--interpolate", action='store_true')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--multi_gpu", action='store_true')
parser.add_argument("--root", type=str, default='~/dataset')
args = parser.parse_args()

print(args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device ...", device)

down_scale = args.down_scale

train_loader, test_loader = datasets.CIFAR10(args)
num_classes = 10

def get_model(model, pretrained=False, num_classes = 10):
    if model == 'resnet32':
        net = resnet32()
        # net.fc = nn.Linear(in_features=512, out_features=num_classes)
    return net

model_name = args.model_name
net = get_model(model_name, pretrained=args.pretrained, num_classes=num_classes)
net = net.to(device)

if args.multi_gpu == True:
    net = nn.DataParallel(net)

optimizer = optim.SGD(net.parameters(),momentum=0.9,lr=0.1, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,150])

def train(train_loader):
    criterion = nn.CrossEntropyLoss()
    
    
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
    
    scheduler.step()

    train_accuracy = n_corrects/n_count
    train_avg_loss /= n_count

    return train_accuracy, train_avg_loss, net

def evaluate(test_loader):
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

def train_and_eval(epochs, train_loader, test_loader, save_name='default.pth'):
    print("─── Start Training & Evalutation ───")
    
    best_accuracy = 0
    best_model = None
    
    for i in range(epochs):
        time_start = time.time()
        
        print(f"┌── Epoch ({i}/{epochs-1})")
        
        train_acc, loss, net = train(train_loader)
        print(f"├── Training Loss : {loss:.4f}")
        print(f'├── Training accuracy : {train_acc*100:.2f}%')
        print("│")
        test_acc, net = evaluate(test_loader)
        print(f'└── Testing accuracy : {test_acc*100:.2f}%')
        
        if test_acc > best_accuracy:
            print(f"  └──> Saving the best model to \"{save_name}\"")
            best_accuracy = test_acc
            if args.multi_gpu == True:
                best_model = net.module.state_dict()
            else:
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

accuracy, net = train_and_eval(epochs, train_loader, test_loader, save_name=f"./models/CIFAR10/best_{model_name}_x{down_scale}_{args.pretrained}.pth")

print(f"Done with accuracy : {accuracy*100:.2f}%")




