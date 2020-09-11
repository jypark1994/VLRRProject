import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MultiscaleDataset(Dataset):
    def __init__(self, root, transform, LR_scale=2, HR_size=[224, 224], keep_size=False):
        self.image_list = []
        self.LR_scale = LR_scale

        self.transform = transform

        # DFD Requires same LR, HR size. (Interpolated LR)
        if keep_size == True:
            # Couldn't expect computation reduction.
            self.LR_transform = transforms.Compose([
                transforms.Resize(HR_size[0]//LR_scale,
                                  interpolation=Image.BICUBIC),
                transforms.Resize(HR_size, interpolation=Image.BICUBIC)
            ])
        else:
            self.LR_transform = transforms.Resize(HR_size[0]//LR_scale)

        self.ToTensor = transforms.ToTensor()
        # self.Normalizer = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        class_dirs = os.listdir(root)

        for label, class_dir in enumerate(class_dirs):
            image_dirs = os.listdir(os.path.join(root, class_dir))
            for img_path in image_dirs:
                self.image_list.append(
                    [label, os.path.join(root, class_dir, img_path)])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        label = self.image_list[index][0]
        img = Image.open(self.image_list[index][1])

        if self.transform != None:
            img = self.transform(img)

        HR = self.ToTensor(img)
        LR = self.ToTensor(self.LR_transform(img))

        # Exception handling for the grayscale images.
        if(HR.shape[0] == 1):
            HR = HR.repeat(3, 1, 1)
        if(LR.shape[0] == 1):
            LR = LR.repeat(3, 1, 1)
        # ----------------------------------------------

        # HR = self.Normalizer(HR)
        # LR = self.Normalizer(LR)

        return LR, HR, label

def get_train_loader(root, LR_scale=4, HR_size=[32, 32], batch_size=128, num_workers=8, keep_size=False):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    data_path = os.path.join(root, 'CIFAR10_RAW/train')
    train_dataset = MultiscaleDataset(
        data_path, transform=transform, LR_scale=LR_scale, HR_size=HR_size, keep_size=keep_size)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader


def get_test_loader(root, LR_scale=4, HR_size=[32, 32], batch_size=128, num_workers=8, keep_size=False):
    data_path = os.path.join(root, 'CIFAR10_RAW/test')
    test_dataset = MultiscaleDataset(
        data_path, transform=None, LR_scale=LR_scale, HR_size=HR_size, keep_size=keep_size)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return test_dataloader

def MultiScaleCIFAR10(root, LR_scale, batch_size):
    train_loader = get_train_loader(root=root, LR_scale=LR_scale,
                                    batch_size=batch_size, num_workers=8, keep_size=True)
    test_loader = get_test_loader(root=root, LR_scale=LR_scale,
                                batch_size=batch_size, num_workers=8, keep_size=True)
    return train_loader, test_loader