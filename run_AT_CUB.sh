python3 ./Attention_Transfer.py --expr_name "ATT_L4_MEAN_CUB_A05_B1_T5_B128_LR01_WD0" --teacher_weight "./pretrained/CUB200/CUB200_ResNet34_HR.pth" \
        --gpus 0 --root "../dataset/" --n_epochs 300 --alpha 0.5 --beta 1 --temperature 5 --learning_rate 0.1 --weight_decay 0 \
        --batch_size 128 --img_type "cub200" --num_classes 200 --target_idx -1 --attention_metric mean \
|
python3 ./Attention_Transfer.py --expr_name "ATT_L4_MEAN_CUB_A05_B1_T5_B128_LR01_WD1e-4" --teacher_weight "./pretrained/CUB200/CUB200_ResNet34_HR.pth" \
        --gpus 1 --root "../dataset/" --n_epochs 300 --alpha 0.5 --beta 1 --temperature 5 --learning_rate 0.1 --weight_decay 1e-4 \
        --batch_size 128 --img_type "cub200" --num_classes 200 --target_idx -1 --attention_metric mean;

python3 ./Attention_Transfer.py --expr_name "ATT_L4_MEAN_CUB_A09_B1_T5_B128_LR01_WD0" --teacher_weight "./pretrained/CUB200/CUB200_ResNet34_HR.pth" \
        --gpus 0 --root "../dataset/" --n_epochs 300 --alpha 0.9 --beta 1 --temperature 5 --learning_rate 0.1 --weight_decay 0 \
        --batch_size 128 --img_type "cub200" --num_classes 200 --target_idx -1 --attention_metric mean \
|
python3 ./Attention_Transfer.py --expr_name "ATT_L4_MEAN_CUB_A09_B1_T5_B128_LR01_WD1e-4" --teacher_weight "./pretrained/CUB200/CUB200_ResNet34_HR.pth" \
        --gpus 1 --root "../dataset/" --n_epochs 300 --alpha 0.9 --beta 1 --temperature 5 --learning_rate 0.1 --weight_decay 1e-4 \
        --batch_size 128 --img_type "cub200" --num_classes 200 --target_idx -1 --attention_metric mean;