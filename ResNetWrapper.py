import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from model_cifar.resnet_cifar import resnet32
# class EnhanceNet(nn.Module):
#     def __init__(self, in_planes, hidden_planes, out_planes):
#         super(EnhanceNet, self).__init__()
#         self.conv3x3 = nn.Conv2d(in_planes,hidden_planes,kernel_size=3, stride=1, padding=1, groups=in_planes)
#         self.conv5x5 = nn.Conv2d(in_planes,hidden_planes,kernel_size=5, stride=1, padding=1, groups=in_planes)
#         self.conv7x7 = nn.Conv2d(in_planes,hidden_planes,kernel_size=7, stride=1, padding=1, groups=in_planes)
#         self.conv1x1 = nn.Conv2d(hidden_planes*3, out_planes, kernel_size=1, stride=1, padding=0)


#     def forward(self, x):
#         f_33 = self.conv3x3(x)
#         f_55 = self.conv5x5(x)
#         f_77 = self.conv7x7(x)

#         f_cat = torch.cat((f_33, f_55, f_77), dim=1)
#         f_fuse = self.conv1x1(f_cat) 

#         return f_fuse


class ResNetWrapper(nn.Module):
    '''
        ResNet model wrapper for pytorch's official implementation
        - Classifying ImageNet-sized(224x224x3) images.
        Junyoung Park : jy_park@inu.ac.kr
    '''
    def __init__(self, net, n_classes, mode='imagenet', pretrained_weight=None):
        super(ResNetWrapper, self).__init__()
        self.net = net
        self.n_classes = n_classes

        if mode.lower() == 'imagenet':
            print("Wrapper mode : ResNet-ImageNet")
            if self.n_classes != self.net.fc.out_features:
                print(f"- Out channels : {n_classes}")
                self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)
            self._forward = self._forward_imagenet # Use net.fc as FC

        else: # if mode == cifar
            print("Wrapper mode : ResNet-CIFAR")
            if self.n_classes != self.net.linear.out_features:
                print(f"- Out channels : {n_classes}")
                self.net.linear = nn.Linear(self.net.linear.in_features, n_classes)
            self._forward = self._forward_cifar # Use net.linear as FC

        if pretrained_weight != None:
            self._load_pretrained_weight(pretrained_weight)

    def _load_pretrained_weight(self, w_p):
        state_dict = torch.load(w_p)
        print(f"Load state dict with accuracy : {state_dict['acc']*100:.2f}%")
        print(f"- Weight dir : {w_p}")
        self.net.load_state_dict(state_dict['net'])

    def _forward_cifar(self, x):
        C1 = F.relu(self.net.bn1(self.net.conv1(x)))
        L1 = self.net.layer1(C1)
        L2 = self.net.layer2(L1)
        L3 = self.net.layer3(L2)
        f = F.avg_pool2d(L3, L3.size()[3])
        f = f.view(f.size(0), -1)
        out = self.net.linear(f)
        return L3, out

    def _forward_imagenet(self, x):
        C1 = self.net.relu(self.net.bn1(self.net.conv1(x)))
        C1_p = self.net.maxpool(C1)
        L1 = self.net.layer1(C1_p)
        L2 = self.net.layer2(L1)
        L3 = self.net.layer3(L2)
        L4 = self.net.layer4(L3)
        f = self.net.avgpool(L4)
        f = torch.flatten(f, 1)
        out = self.net.fc(f)
        return L4, out

    def forward(self, x):
        return self._forward(x)