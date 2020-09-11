python3 CIFAR_DIST_REFACTOR.py --batch_size 64 --alpha 0.9 --temperature 2 \
        --expr_name "RESNET34_A09_T2_LR1E-2_BS64" --gpus "0" --teacher_weight "./pretrained/CIFAR10/ResNet34_HR.pth" > RESNET34_A09_T2_LR1E-2_BS64.txt |
python3 CIFAR_DIST_REFACTOR.py --batch_size 64 --alpha 0.9 --temperature 5 \
        --expr_name "RESNET34_A09_T5_LR1E-2_BS64" --gpus "1" --teacher_weight "./pretrained/CIFAR10/ResNet34_HR.pth" > RESNET34_A09_T5_LR1E-2_BS64.txt |
python3 CIFAR_DIST_REFACTOR.py --batch_size 64 --alpha 0.5 --temperature 10 \
        --expr_name "RESNET34_A05_T10_LR1E-2_BS64" --gpus "2" --teacher_weight "./pretrained/CIFAR10/ResNet34_HR.pth" > RESNET34_A05_T10_LR1E-2_BS64.txt |
python3 CIFAR_DIST_REFACTOR.py --batch_size 64 --alpha 0.5 --temperature 20 \
        --expr_name "RESNET34_A05_T20_LR1E-2_BS64" --gpus "3" --teacher_weight "./pretrained/CIFAR10/ResNet34_HR.pth" > RESNET34_A05_T20_LR1E-2_BS64.txt;