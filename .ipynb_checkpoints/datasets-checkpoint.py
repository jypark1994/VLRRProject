from torch import utils
from torchvision import datasets, transforms

# args : batch_size, num_workers, down_scale

def ILSVRC_Birds(args):

    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224//args.down_scale),
        transforms.ToTensor(),
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(224//args.down_scale),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder("../dataset/ILSVRC_Birds/train", transform=transforms_train)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = datasets.ImageFolder("../dataset/ILSVRC_Birds/val", transform=transforms_test)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    return train_loader, test_loader
    
def MosquitoDL(args):
    
    init_scale = 1.15
    
    transforms_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1),
        transforms.RandomAffine(360,scale=[init_scale-0.15,init_scale+0.15]),
        transforms.CenterCrop(224),
        transforms.Resize(224//args.down_scale),
        transforms.ToTensor()
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(224//args.down_scale),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder("/media/data/MosquitoDL/TrainVal", transform=transforms_train)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = datasets.ImageFolder("/media/data/MosquitoDL/Test", transform=transforms_test)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    return train_loader, test_loader

def MosquitoMultiscale(args):
    
    init_scale = 1.15
    
    transforms_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1),
        transforms.RandomAffine(360,scale=[init_scale-0.15,init_scale+0.15]),
        transforms.CenterCrop(224),
        transforms.Resize(224//args.down_scale),
        transforms.ToTensor()
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(224//args.down_scale),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder("/media/data/MosquitoDL/TrainVal", transform=transforms_train)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = datasets.ImageFolder("/media/data/MosquitoDL/Test", transform=transforms_test)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    return train_loader, test_loader