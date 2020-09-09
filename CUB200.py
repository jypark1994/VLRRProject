import os
import pandas as pd
import PIL.Image as Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

class CUB2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, LR_scale=None, transform_norm=None, loader=default_loader, \
            download=True, keep_size=True, HR_size=[224,224]):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.transform_norm = transform_norm
        self.LR_scale = LR_scale
        self.ToTensor = transforms.ToTensor()

        if LR_scale > 1: # Want to use LR images?
            if keep_size == True: # Some methods require same LR/HR size. (Interpolated LR)
                self.LR_transform = transforms.Compose([
                    transforms.Resize(HR_size[0]//LR_scale),
                    transforms.Resize(HR_size, interpolation=Image.BICUBIC)
                ])
            else:
                self.LR_transform = transforms.Resize(HR_size[0]//LR_scale)

        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None: # If transform required.
            img = self.transform(img)

        if self.LR_scale > 1: # If LR scale required.
            img = self.LR_transform(img)
        
        img = self.ToTensor(img)

        if self.transform_norm is not None: # If normalization required.
            img = self.transform_norm(img)

        return img, target

class CUB2011_Multiscale(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, transform_norm=None, loader=default_loader, download=True, \
             LR_scale=2, HR_size=[224,224], keep_size=False):
        self.root = os.path.expanduser(root)
        if LR_scale > 1:
            if keep_size == True: # DFD Requires same LR, HR size. (Interpolated LR)
                # Couldn't expect computation reduction.
                self.LR_transform = transforms.Compose([
                    transforms.Resize(HR_size[0]//LR_scale),
                    transforms.Resize(HR_size, interpolation=Image.BICUBIC)
                ])
            else:
                self.LR_transform = transforms.Resize(HR_size[0]//LR_scale)

        self.ToTensor = transforms.ToTensor()
        self.transform = transform
        self.transform_norm = transform_norm
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        
        HR = self.ToTensor(img)
        LR = self.ToTensor(self.LR_transform(img))
        
        if self.transform_norm is not None:
            HR = self.transform_norm(HR)
            LR = self.transform_norm(LR)

        return HR, LR, target

transforms_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(45),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
])
transforms_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])
transforms_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def CUB200_MultiScale_get_loaders(root="~/dataset/",batch_size=32, num_workers=8, LR_scale=4, keep_size=True):
    train_dataset = CUB2011_Multiscale(root=root, train=True, transform=transforms_train, \
                transform_norm=transforms_norm, LR_scale=LR_scale, keep_size=keep_size)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    test_dataset = CUB2011_Multiscale(root=root, train=False, transform=transforms_test, \
                transform_norm=transforms_norm, LR_scale=LR_scale, keep_size=keep_size)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    return train_loader, test_loader

def CUB200_get_loaders(root="~/dataset/",batch_size=32, num_workers=8, LR_scale=4, keep_size=True):
    train_dataset = CUB2011(root=root, train=True, transform=transforms_train, transform_norm=transforms_norm, LR_scale=LR_scale, keep_size=keep_size)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    test_dataset = CUB2011(root=root, train=False, transform=transforms_test, transform_norm=transforms_norm, LR_scale=LR_scale, keep_size=keep_size)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    return train_loader, test_loader

