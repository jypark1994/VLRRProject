python3 ./Distillation.py --expr_name "DIST_KL_L4_CIFAR_A09_B0_T4_B128_L1E-1_WD5E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet32_92_47.pth" \
        --gpus 0 --root "../dataset/" --n_epochs 300 --alpha 0.9 --temperature 4 --learning_rate 1e-1 --weight_decay 0 --decay_step 60 \
        --kd_loss "KL" --batch_size 128 --img_type "cifar10" --num_classes 10 --arch_teacher resnet32 --arch_student resnet32 \
        > DIST_KL_L4_CIFAR_A09_B0_T4_B128_L1E-1_WD5E-4.txt \
|
python3 ./Distillation.py --expr_name "DIST_KL_L4_CIFAR_A07_B0_T4_B128_L1E-1_WD5E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet32_92_47.pth" \
        --gpus 1 --root "../dataset/" --n_epochs 300 --alpha 0.7 --temperature 4 --learning_rate 1e-1 --weight_decay 0 --decay_step 60 \
        --kd_loss "KL" --batch_size 128 --img_type "cifar10" --num_classes 10 --arch_teacher resnet32 --arch_student resnet32 \
        > DIST_KL_L4_CIFAR_A07_B0_T4_B128_L1E-1_WD5E-4.txt \
|
python3 ./Distillation.py --expr_name "DIST_KL_L4_CIFAR_A05_B0_T4_B128_L1E-1_WD5E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet32_92_47.pth" \
        --gpus 2 --root "../dataset/" --n_epochs 300 --alpha 0.5 --temperature 4 --learning_rate 1e-1 --weight_decay 0 --decay_step 60 \
        --kd_loss "KL" --batch_size 128 --img_type "cifar10" --num_classes 10 --arch_teacher resnet32 --arch_student resnet32 \
        > DIST_KL_L4_CIFAR_A05_B0_T4_B128_L1E-1_WD5E-4.txt \
|
python3 ./Distillation.py --expr_name "DIST_KL_L4_CIFAR_A03_B0_T4_B128_L1E-1_WD5E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet32_92_47.pth" \
        --gpus 3 --root "../dataset/" --n_epochs 300 --alpha 0.3 --temperature 4 --learning_rate 1e-1 --weight_decay 0 --decay_step 60 \
        --kd_loss "KL" --batch_size 128 --img_type "cifar10" --num_classes 10 --arch_teacher resnet32 --arch_student resnet32 \
        > DIST_KL_L4_CIFAR_A03_B0_T4_B128_L1E-1_WD5E-4.txt;

python3 ./Distillation.py --expr_name "DIST_SCRATCH_KL_L4_CIFAR_A09_B0_T4_B128_L1E-1_WD5E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet32_92_47.pth" \
        --scratch_student --gpus 0 --root "../dataset/" --n_epochs 300 --alpha 0.9 --temperature 4 --learning_rate 1e-1 --weight_decay 0 --decay_step 60 \
        --kd_loss "KL" --batch_size 128 --img_type "cifar10" --num_classes 10 --arch_teacher resnet32 --arch_student resnet32 \
        > DIST_SCRATCH_KL_L4_CIFAR_A09_B0_T4_B128_L1E-1_WD5E-4.txt \
|
python3 ./Distillation.py --expr_name "DIST_SCRATCH_KL_L4_CIFAR_A07_B0_T4_B128_L1E-1_WD5E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet32_92_47.pth" \
        --scratch_student --gpus 1 --root "../dataset/" --n_epochs 300 --alpha 0.7 --temperature 4 --learning_rate 1e-1 --weight_decay 0 --decay_step 60 \
        --kd_loss "KL" --batch_size 128 --img_type "cifar10" --num_classes 10 --arch_teacher resnet32 --arch_student resnet32 \
        > DIST_SCRATCH_KL_L4_CIFAR_A07_B0_T4_B128_L1E-1_WD5E-4.txt \
|
python3 ./Distillation.py --expr_name "DIST_SCRATCH_KL_L4_CIFAR_A05_B0_T4_B128_L1E-1_WD5E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet32_92_47.pth" \
        --scratch_student --gpus 2 --root "../dataset/" --n_epochs 300 --alpha 0.5 --temperature 4 --learning_rate 1e-1 --weight_decay 0 --decay_step 60 \
        --kd_loss "KL" --batch_size 128 --img_type "cifar10" --num_classes 10 --arch_teacher resnet32 --arch_student resnet32 \
        > DIST_SCRATCH_KL_L4_CIFAR_A05_B0_T4_B128_L1E-1_WD5E-4.txt \
|
python3 ./Distillation.py --expr_name "DIST_SCRATCH_KL_L4_CIFAR_A03_B0_T4_B128_L1E-1_WD5E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet32_92_47.pth" \
        --scratch_student --gpus 3 --root "../dataset/" --n_epochs 300 --alpha 0.3 --temperature 4 --learning_rate 1e-1 --weight_decay 0 --decay_step 60 \
        --kd_loss "KL" --batch_size 128 --img_type "cifar10" --num_classes 10 --arch_teacher resnet32 --arch_student resnet32 \
        > DIST_SCRATCH_KL_L4_CIFAR_A03_B0_T4_B128_L1E-1_WD5E-4.txt;