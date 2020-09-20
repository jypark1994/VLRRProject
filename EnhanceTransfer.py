import torch
import torch.nn.functional as F
from ResNetWrapper import ResNetWrapper
from torch import nn, optim
from torchvision.utils import make_grid

import os
import shutil
import time
import argparse
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--expr_name", type=str, default="Default")
parser.add_argument("--teacher_weight", type=str,
                    help="Pretrained teacher dir (*.pth)")
parser.add_argument("--gpus", type=str, default="",
                    help="If blank, use CPU computation.")
parser.add_argument("--root", type=str, default="/home/ryan/dataset")
parser.add_argument("--n_epochs", type=int, default=300)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--temperature", type=float, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=4e-5)
parser.add_argument("--LR_scale", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--show_batch", action="store_true", default=False)
parser.add_argument("--img_type", type=str, default="cub200")
parser.add_argument("--num_classes", type=int, default=200)
parser.add_argument("--target_idx", type=int, default=-1)
args = parser.parse_args()

# Init training environment.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
plt.rcParams['figure.figsize'] = (30, 30)


# Define and make experiment directory.
data_name = args.img_type
expr_path = f"./experiments/{data_name}/{args.expr_name}"

if(os.path.isdir(expr_path)):
    print("Delete last experiment results")
    shutil.rmtree(expr_path)
else:
    print("Make new experiment folder")
    os.makedirs(expr_path, exist_ok=True)
writer = SummaryWriter(logdir=os.path.join(expr_path, "runs"))

writer.add_text("Settings", str(args))

# Define Train/Test dataloaders.
if data_name == 'cifar10':
    from MultiScaleDatasets import MultiScaleCIFAR10
    train_loader, test_loader = MultiScaleCIFAR10(
        args.root, args.LR_scale, args.batch_size)
elif data_name == 'cub200':
    from CUB200 import CUB200_MultiScale_get_loaders
    train_loader, test_loader = CUB200_MultiScale_get_loaders(
        args.root, batch_size=args.batch_size)
else:
    print(f"Unsupported Dataset : {data_name}")

# Plot batch samples.
if args.show_batch:
    for data in train_loader:
        LR, HR, label = data[0], data[1], data[2]
        print(HR.shape, LR.shape, label.shape)

        fig, ax = plt.subplots(1, 2)

        LR_samples = make_grid(LR, nrow=8).permute(1, 2, 0)
        HR_samples = make_grid(HR, nrow=8).permute(1, 2, 0)
        ax[0].imshow(HR_samples)
        ax[0].axis('off')
        ax[0].set_title('HR Samples')
        ax[1].imshow(LR_samples)
        ax[1].axis('off')
        ax[1].set_title('LR Samples')
        plt.show()
        break

    for data in test_loader:
        LR, HR, label = data[0], data[1], data[2]
        print(HR.shape, LR.shape, label.shape)

        fig, ax = plt.subplots(1, 2)

        LR_samples = make_grid(LR, nrow=8).permute(1, 2, 0)
        HR_samples = make_grid(HR, nrow=8).permute(1, 2, 0)
        ax[0].imshow(HR_samples)
        ax[0].axis('off')
        ax[0].set_title('HR Samples')
        ax[1].imshow(LR_samples)
        ax[1].axis('off')
        ax[1].set_title('LR Samples')
        plt.show()
        break

# Get model


def get_model(model, n_classes, data_name, pretrained_dir):
    if data_name == 'cifar10':  # 32x32x3 input
        import models_cifar
        if model == 'resnet50':
            net = models_cifar.resnet.ResNet50()
        elif model == 'resnet34':
            net = models_cifar.resnet.ResNet34()

        net = ResNetWrapper(net, 10, mode='cifar',
                            pretrained_weight=pretrained_dir)
    else:  # The others (224x224x3)
        from torchvision import models
        if model == 'resnet50':
            net = models.resnet50(pretrained=True)
        elif model == 'resnet34':
            net = models.resnet34(pretrained=True)

        net = ResNetWrapper(net, n_classes, mode='imagenet',
                            pretrained_weight=pretrained_dir)
    return net

# Define models and load pretrained weights.


device = 'cuda' if torch.cuda.is_available() else 'cpu'

teacher_name = 'resnet34'
student_name = 'resnet34'
print(f"Teacher: {teacher_name}, Student: {student_name}")

net_t = get_model(teacher_name, args.num_classes, data_name, args.teacher_weight)
net_t = net_t.to(device)

net_s = get_model(student_name, args.num_classes, data_name, args.teacher_weight)
net_s = net_s.to(device)

net_t = nn.DataParallel(net_t)
net_s = nn.DataParallel(net_s)


class KD_loss():
    def __init__(self, params):
        self.fn = nn.KLDivLoss()
        self.alpha = params['alpha']
        self.T = params['T']

    def __call__(self, student_outputs, teacher_outputs):
        return self.fn(F.log_softmax(student_outputs/self.T, dim=1),
                       F.softmax(teacher_outputs/self.T, dim=1)) * (self.alpha * self.T * self.T)


class Attention_loss():
    '''
        Acquiring attention loss between two feature maps.

        [Usage]
            criterion : Distance metric (e.g. MSE, MAE ...)
            target_idx : Target layers (e.g. target_idx = -1 for the last layer 'resnet.layer4')
            - 'Out of Bound' warning !
    '''
    def __init__(self, criterion, target_idx=-1):
        self.criterion = criterion
        self.target_idx = target_idx

    def __call__(self, student_features, teacher_features):
        loss = 0
        len_features = len(student_features)

        zipped_features = zip(student_features[:self.target_idx], teacher_features[:self.target_idx])

        for f_t, f_s in zipped_features:
            loss += self.criterion(f_t, f_s)
        return loss / len_features


def distillate(teacher, student, train_loader, cur_epoch):
    params = {'alpha': args.alpha, 'beta': args.beta, 'T': args.temperature}

    criterion_CEL = nn.CrossEntropyLoss()
    criterion_KLD = KD_loss(params)
    criterion_ATT = Attention_loss(nn.MSELoss(),args.target_idx)

    # DFD uses SGD with momentum 0.9, weight deacy 5e-4.
    optimizer = optim.SGD(student.parameters(
    ), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)

    train_avg_loss = 0
    n_count = 0
    n_corrects = 0

    teacher.eval()
    teacher.requires_grad = False
    student.train()

    for i, data in enumerate(train_loader):
        data_len = len(train_loader)

        batch_lr, batch_hr, label = data[0].to(
            device), data[1].to(device), data[2].to(device)

        optimizer.zero_grad()

        features_t, pred_t = teacher(batch_hr)
        features_s, pred_s = student(batch_lr)

        soft_loss = (1.-params['alpha'])*criterion_KLD(pred_s, pred_t)
        data_loss = params['alpha']*criterion_CEL(pred_s, label)
        att_loss = params['beta']*criterion_ATT(features_t, features_s)

        loss = soft_loss + data_loss + att_loss
        train_avg_loss += loss

        writer.add_scalar("loss/KD", soft_loss,
                          global_step=i+data_len*cur_epoch)
        writer.add_scalar("loss/CE", data_loss,
                          global_step=i+data_len*cur_epoch)
        writer.add_scalar("loss/ATT", att_loss,
                          global_step=i+data_len*cur_epoch)
        writer.add_scalar("loss/Total", loss,
                          global_step=i+data_len*cur_epoch)

        batch_correct = torch.sum(torch.argmax(pred_s, dim=1) == label).item()
        batch_count = label.shape[0]

        batch_acc = batch_correct/batch_count*100
        writer.add_scalar("acc/train", batch_acc,
                          global_step=i+data_len*cur_epoch)

        n_corrects += batch_correct
        n_count += batch_count

        loss.backward()
        optimizer.step()

        if(i == len(train_loader)-1):
            print(
                f"Soft loss = {soft_loss:.4f}, Hard loss = {data_loss:.4f}, Att loss = {att_loss:.4f}, Total loss = {loss:.4f}")

    train_accuracy = n_corrects/n_count
    train_avg_loss /= n_count

    return train_accuracy, train_avg_loss, student


def evaluate(net, test_loader, cur_epoch, eval_target='LR'):
    net.eval()

    n_count = 0
    n_corrects = 0

    if eval_target == 'LR':
        target = 0
    else:
        target = 1

    for j, data in enumerate(test_loader):
        data_len = len(test_loader)
        batch, label = data[target].to(device), data[2].to(device)

        _, pred = net(batch)

        batch_correct = torch.sum(torch.argmax(pred, dim=1) == label).item()
        batch_count = label.shape[0]

        batch_acc = batch_correct/batch_count*100
        writer.add_scalar("acc/test", batch_acc, j+data_len*cur_epoch)

        n_corrects += batch_correct
        n_count += batch_count

    test_accuracy = n_corrects/n_count

    return test_accuracy, net


def train_and_eval(teacher, student, epochs, train_loader, test_loader, save_name='default.pth'):
    print("─── Start Training & Evalutation ───")

    best_accuracy = 0
    best_model = None

    for i in range(epochs):
        time_start = time.time()
        print(f"┌ Epoch ({i}/{epochs-1})")

        train_acc, loss, student = distillate(
            teacher, student, train_loader, i)
        print(f"├── Training Loss : {loss:.4f}")
        print(f'├── Training accuracy : {train_acc*100:.2f}%')
        print("│ Testing ...")
        test_acc, student = evaluate(student, test_loader, i, eval_target='LR')
        print(f'└── Testing accuracy : {test_acc*100:.2f}%')

        if test_acc > best_accuracy:
            print(f"  └──> Saving the best model to \"{save_name}\"")
            best_accuracy = test_acc
            writer.add_text("BestAcc", f"{best_accuracy*100:.2f}% @ epoch {i}")
            best_model = student
            model_dict = {'acc': best_accuracy, 'net': best_model}
            torch.save(model_dict, save_name)

        time_end = time.time()

        epoch_time = time_end - time_start
        epoch_time_gm = time.gmtime(epoch_time)
        estimated_time = epoch_time * (epochs - 1 - i)
        estimated_time_gm = time.gmtime(estimated_time)
        print(
            f"Epoch time ─ {epoch_time_gm.tm_hour}[h] {epoch_time_gm.tm_min}[m] {epoch_time_gm.tm_sec}[s]")
        print(
            f"Estimated time ─ {estimated_time_gm.tm_hour}[h] {estimated_time_gm.tm_min}[m] {estimated_time_gm.tm_sec}[s]")
        print("\n")

    return best_accuracy, best_model


epochs = args.n_epochs

accuracy, net_t = train_and_eval(net_t, net_s, epochs, train_loader, test_loader,
                                 save_name=f"{expr_path}/T{teacher_name}_S{student_name}_x{args.LR_scale}.pth")
