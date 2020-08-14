#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# Make the model classifies LR images well !
# 
# # How?
# 
# Adapt SR models using multiple loss function including Attention, Soft label

# In[1]:


import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim, utils
from torchvision import models, datasets, transforms

import PIL.Image as Image

from tqdm import tqdm
import random
import cv2
import os

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10,10)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device ...", device)


# # Defining Dataset
# 
# - 'Dataset' should return GT_image and Labels

# In[2]:


class MosquitoDatasets(utils.data.Dataset):
    def __init__(self, root, transforms, scale=4):
        # Root should be a specific dataset path (../train or ../eval)
        super(MosquitoDatasets, self).__init__()
        
        self.data = []
        self.scale = scale
        
        self.classes = os.listdir(root)
        self.classes.sort()
        
        self.transforms = transforms
        
        for idx, cls in enumerate(self.classes):
            cls_path = os.path.join(root, cls)
            images = os.listdir(cls_path)
            for img in images:
                image_path = os.path.join(cls_path, img)
                self.data.append((idx, image_path))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.data[idx][0]
        image = self.data[idx][1]
        
        gt_image = Image.open(image)
        output = self.transforms(gt_image)
#         gt_size = gt_image.shape
#         lr_image = cv2.resize(gt_image, (gt_size[0]//self.scale, gt_size[1]//self.scale),cv2.INTER_CUBIC)
        return (output, label)

init_scale = 1.15

transforms_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1),
    transforms.RandomAffine(360,scale=[init_scale-0.15,init_scale+0.15]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
    
# dataset = MosquitoDatasets("/data/MosquitoDL/TrainVal/", transforms=mos_transforms)
dataset = datasets.ImageFolder("/data/MosquitoDL/TrainVal/", transform=transforms_train)
dataloader = utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Show samples

batch_sample = next(iter(dataloader))
image_samples = torchvision.utils.make_grid(batch_sample[0],nrow=8).permute((1,2,0))

fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(image_samples)
plt.show()


# # Defining LR Generator

# In[3]:


import numpy

class LRGenerator(object): # Follows PyTorch custom transformer
    def __init__(self, scale, keep_size, device):
        # Options will be specified
        self.scale = scale
        self.keep_size = keep_size
        self.device = device

        
    def generate(self, x):
        x = x.detach().cpu().float()
        
        N, C, W, H = x.size()
#         print(N, W, H, C)
        
        lr_image = torch.zeros([N,C,W,H])
        
        # Iterate for each image in a batch.
        for i, d in enumerate(x):
            # Randomly select a specific scale - Batch-wise
            _scale = random.choice(self.scale)

            # Define Down/Upscaling Transformers
            tf_down = transforms.Resize((W//_scale, H//_scale))
            tf_up = transforms.Resize((W, H), interpolation=Image.BICUBIC)
            
            x_pil = transforms.ToPILImage()(d)
            
            x_rescaled = tf_down(x_pil)
            
            if self.keep_size == True:
                x_rescaled = tf_up(x_rescaled)
                
            x_rescaled = transforms.ToTensor()(x_rescaled)
            
            lr_image[i] = x_rescaled.clone()
            
        return lr_image.to(device)


# # Defining the Model

# In[4]:


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# In[5]:


def modified_mobilenetv2():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[-1] = nn.Linear(in_features=1280, out_features=6)
    return model

def modified_vgg16():
    model = models.vgg16(pretrained=True)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=6)
    return model


# In[6]:


class EnhancedClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EnhancedClassifier, self).__init__()
        
        self.upscaler = SRCNN(num_channels=3)
        
        # Working as adaptive data augmentation for downsampling
        self.lr_generator = LRGenerator(scale=[2,4,8], keep_size=True, device='cuda')
        
        self.classifier = modified_mobilenetv2()
        
    def forward(self, HR):
        LR = self.lr_generator.generate(HR) # Useless for testing
        SR = self.upscaler(LR)
        
#         self.show_samples(LR)
        
        score_HR = self.classifier(HR) # Useless for testing
        score_SR = self.classifier(SR)
        
        
        
        return LR, SR, score_SR, score_HR
    
    def show_samples(self, batch):
        batch_sample = batch.detach().cpu()
        image_samples = torchvision.utils.make_grid(batch_sample,nrow=2).permute((1,2,0))

        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(image_samples)
        plt.show()
        

# x = torch.randn([1,3,224,224])
# model = EnhancedClassifier(n_classes=6)
# print(model(x))


# In[7]:


def loss_fn_kd(score_SR, labels, score_GT, params): 
    # Original Implimentation : peterlint (GitHub)
    """ 
    Compute the knowledge-distillation (KD) loss given outputs, labels. 
    "Hyperparameters": temperature and alpha 

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher 
    and student expects the input tensor to be log probabilities! See Issue #2 
    """ 
    
    alpha = params['alpha'] # Bias between softmax and label loss.
    T = params['temperature'] # The amount of softening scores.
    
    KD_loss = nn.KLDivLoss()(F.log_softmax(score_SR/T, dim=1),
                             F.softmax(score_GT/T, dim=1)) * (alpha * T * T) + F.cross_entropy(score_SR, labels) * (1. - alpha) 
  
    return KD_loss 


# # Train the Model
# 
# ### Loss Functions
# 
# - Hard Loss : 모델이 원본 이미지를 정확히 분류하도록 유도.
# - Soft Loss : 모델이 SR 이미지를 '원본 이미지 처럼' 정확히 분류하도록 유도.
# - Image Loss : LR 이미지가 '분류하기 좋게' SR 되도록 유도.
# 
# ### Evaluation Criterion
# 
# - SR 이미지 분류 정확도를 평가하자 !

# In[ ]:


model = EnhancedClassifier(n_classes=6)

model.to(device)

optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=4e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)

loss1 = nn.L1Loss()

lambda_1 = 1

epochs = 100

params = {
    'alpha':0.5,
    'temperature':0.5
}

for n in range(epochs):
    n_corrects = 0
    average_loss = 0
    print(f"[{n}/{epochs}] Current LR : {optimizer.param_groups[0]['lr']}")
    
    for i, data in enumerate(dataloader):

        GT, label = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        LR, SR, score_SR, score_HR = model(GT)
        
        n_corrects += torch.sum(label == torch.argmax(score_SR, dim=1)).item()
        
        image_loss = lambda_1 * loss1(SR, GT)
        kd_loss = loss_fn_kd(score_SR, label, score_HR, params)
        
        if i%100 == 0:
            print(f"Img : {image_loss:.6f}, KD : {kd_loss:.6f}")
        
        multi_loss = image_loss + kd_loss
        average_loss += multi_loss

        multi_loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    average_loss /= len(dataloader)
    accuracy = (n_corrects/(len(dataloader)*32))
    
    print(f"-> Average Loss:{average_loss:.2f}, Accuracy:{accuracy*100:.2f}%")
    
torch.save(model.state_dict(),"./KDSR_200.pth")


# # Evaluation
# 
# - Evaluate the model for each specific scale. (x2, x3, x4)
# 
# - Two base models : LR trained (x2, x3, x4) / GT trained
# 
# - How the KDSR accuracy far from the baselines?

# In[ ]:


test_scale = 4

test_transforms = transforms.Compose([
    transforms.Resize(224//test_scale),
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder('/data/MosquitoDL/Test/',transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 16, shuffle=True, num_workers=8)

model.eval()

n_corrects = 0

for i, data in enumerate(test_loader):

        input_img, label = data[0].to(device), data[1].to(device)

        LR, SR, score_SR, score_HR = model(input_img)
        
        if i == 0:
            print("Printing the first batch images")
            
            image_samples = torchvision.utils.make_grid(input_img.detach().cpu(),nrow=4).permute((1,2,0))
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow((image_samples*255.).int())
            ax.set_title("LR Samples (First Batch)")
            plt.savefig("./LR_samples.jpg")
            
            image_samples = torchvision.utils.make_grid(SR.detach().cpu(),nrow=4).permute((1,2,0))
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow((image_samples*255.).int())
            ax.set_title("SR Samples (First Batch)")
            plt.savefig("./SR_samples.jpg")
            
        n_corrects += torch.sum(label == torch.argmax(score_HR, dim=1)).item()
        
accuracy = (n_corrects/(len(test_loader)*16))
    
print(f"-> Downscaled Accuracy:{accuracy*100:.2f}%")


# In[ ]:




