# python3 ./Attention_Transfer.py --expr_name "ATT_CIFAR_A09_B1_T5_B128_L1E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet34_HR.pth" \
#         --gpus 0 --root "../dataset/" --n_epochs 300 --alpha 0.9 --beta 1 --temperature 5 --learning_rate 1e-4 \
#         --batch_size 128 --img_type "cifar10" --num_classes 10 > ATT_CIFAR_A09_B1_T5_B128_L1E-4.txt \
# |
# python3 ./Attention_Transfer.py --expr_name "ATT_CIFAR_A09_B05_T5_B128_L1E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet34_HR.pth" \
#         --gpus 1 --root "../dataset/" --n_epochs 300 --alpha 0.9 --beta 0.5 --temperature 5 --learning_rate 1e-4 \
#         --batch_size 128 --img_type "cifar10" --num_classes 10 > ATT_CIFAR_A09_B05_T5_B128_L1E-4.txt \
# |
# python3 ./Attention_Transfer.py --expr_name "ATT_CIFAR_A09_B01_T5_B128_L1E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet34_HR.pth" \
#         --gpus 2 --root "../dataset/" --n_epochs 300 --alpha 0.9 --beta 0.1 --temperature 5 --learning_rate 1e-4 \
#         --batch_size 128 --img_type "cifar10" --num_classes 10 > ATT_CIFAR_A09_B01_T5_B128_L1E-4.txt \
# |
# python3 ./Attention_Transfer.py --expr_name "ATT_CIFAR_A09_B15_T5_B128_L1E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet34_HR.pth" \
#         --gpus 3 --root "../dataset/" --n_epochs 300 --alpha 0.9 --beta 1.5 --temperature 5 --learning_rate 1e-4 \
#         --batch_size 128 --img_type "cifar10" --num_classes 10 > ATT_CIFAR_A09_B15_T5_B128_L1E-4.txt;


python3 ./Attention_Transfer.py --expr_name "ATT_CIFAR_A09_B1_T5_B128_L1E-4_WD5E-4" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet34_HR.pth" \
        --gpus 0 --root "../dataset/" --n_epochs 300 --alpha 0.9 --beta 1 --temperature 5 --learning_rate 1e-4 --weight_decay 5e-4 \
        --batch_size 128 --img_type "cifar10" --num_classes 10 > ATT_CIFAR_A09_B1_T5_B128_L1E-4_WD5E-4.txt \
|
python3 ./Attention_Transfer.py --expr_name "ATT_CIFAR_A09_B05_T5_B128_L1E-4_WD1E-3" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet34_HR.pth" \
        --gpus 1 --root "../dataset/" --n_epochs 300 --alpha 0.9 --beta 0.5 --temperature 5 --learning_rate 1e-4 --weight_decay 1e-3 \
        --batch_size 128 --img_type "cifar10" --num_classes 10 > ATT_CIFAR_A09_B05_T5_B128_L1E-4_WD1E-3.txt \
|
python3 ./Attention_Transfer.py --expr_name "ATT_CIFAR_A09_B01_T5_B128_L1E-4_WD5E-3" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet34_HR.pth" \
        --gpus 2 --root "../dataset/" --n_epochs 300 --alpha 0.9 --beta 0.1 --temperature 5 --learning_rate 1e-4 --weight_decay 5e-3 \
        --batch_size 128 --img_type "cifar10" --num_classes 10 > ATT_CIFAR_A09_B01_T5_B128_L1E-4_WD5E-3.txt \
|
python3 ./Attention_Transfer.py --expr_name "ATT_CIFAR_A09_B15_T5_B128_L1E-4_WD1E-2" --teacher_weight "./pretrained/CIFAR10/CIFAR10_ResNet34_HR.pth" \
        --gpus 3 --root "../dataset/" --n_epochs 300 --alpha 0.9 --beta 1.5 --temperature 5 --learning_rate 1e-4 --weight_decay 1e-2 \
        --batch_size 128 --img_type "cifar10" --num_classes 10 > ATT_CIFAR_A09_B15_T5_B128_L1E-4_WD1E-2.txt;