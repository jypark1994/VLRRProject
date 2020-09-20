python3 ./Attention_Transfer.py --expr_name "ATT_MSE_L4_CIFAR_A05_B5_T5_B64_L1E-3_WD0" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet32_HR.pth" \
        --gpus 2 --root "../dataset/" --n_epochs 300 --alpha 0.5 --beta 5 --temperature 5 --learning_rate 1e-1 --weight_decay 0 \
        --batch_size 128 --img_type "cifar10" --num_classes 10 --target_idx -1 --arch_teacher resnet32 --arch_student resnet32 \
|
python3 ./Attention_Transfer.py --expr_name "ATT_MSE_L4_CIFAR_A05_B5_T5_B64_L1E-3_WD1e-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet32_HR.pth" \
        --gpus 3 --root "../dataset/" --n_epochs 300 --alpha 0.5 --beta 5 --temperature 5 --learning_rate 1e-1 --weight_decay 1e-4 \
        --batch_size 128 --img_type "cifar10" --num_classes 10 --target_idx -1 --arch_teacher resnet32 --arch_student resnet32;