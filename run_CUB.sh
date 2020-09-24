# python3 pretrainer_CUB.py --expr_name "ResNet34_Scratch_CUB_x1_L1E-1_W4e-5" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --model_name resnet34 --learning_rate 1e-1 --cuda 0 --weight_decay 4e-5 > "ResNet34_Scratch_CUB_x1_L1E-1_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Scratch_CUB_x4_L1E-1_W4e-5" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --model_name resnet34 --learning_rate 1e-1 --cuda 1 --weight_decay 4e-5 > "ResNet34_Scratch_CUB_x4_L1E-1_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Scratch_CUB_x1_L1E-1_W5e-4" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --model_name resnet34 --learning_rate 1e-1 --cuda 2 --weight_decay 5e-4 > "ResNet34_Scratch_CUB_x1_L1E-1_W5e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Scratch_CUB_x4_L1E-1_W5e-4" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --model_name resnet34 --learning_rate 1e-1 --cuda 3 --weight_decay 5e-4 > "ResNet34_Scratch_CUB_x4_L1E-1_W5e-4.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Scratch_CUB_x1_L1E-1_W1e-4" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --model_name resnet34 --learning_rate 1e-1 --cuda 0 --weight_decay 1e-4 > "ResNet34_Scratch_CUB_x1_L1E-1_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Scratch_CUB_x4_L1E-1_W1e-4" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --model_name resnet34 --learning_rate 1e-1 --cuda 1 --weight_decay 1e-4 > "ResNet34_Scratch_CUB_x4_L1E-1_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Scratch_CUB_x1_L1E-1_W0" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --model_name resnet34 --learning_rate 1e-1 --cuda 2 --weight_decay 5e-4 > "ResNet34_Scratch_CUB_x1_L1E-1_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Scratch_CUB_x4_L1E-1_W0" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --model_name resnet34 --learning_rate 1e-1 --cuda 3 --weight_decay 5e-4 > "ResNet34_Scratch_CUB_x4_L1E-1_W0.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1E-1_W4e-5" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 0 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x1_L1E-1_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1E-1_W4e-5" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 1 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x4_L1E-1_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1E-1_W5e-4" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1E-1_W5e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1E-1_W5e-4" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1E-1_W5e-4.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1E-1_W1e-4" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 0 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x1_L1E-1_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1E-1_W1e-4" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 1 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x4_L1E-1_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1E-1_W0" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1E-1_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1E-1_W0" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1E-1_W0.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-2_W4e-5" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 0 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x1_L1e-2_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-2_W4e-5" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 1 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x4_L1e-2_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-2_W5e-4" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-2_W5e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-2_W5e-4" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-2_W5e-4.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-2_W1e-4" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 0 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x1_L1e-2_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-2_W1e-4" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 1 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x4_L1e-2_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-2_W0" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-2_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-2_W0" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-2_W0.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_W4e-5" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 0 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x1_L1e-3_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_W4e-5" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 1 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x4_L1e-3_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_W5e-4" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-3_W5e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_W5e-4" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-3_W5e-4.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_W1e-4" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 0 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x1_L1e-3_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_W1e-4" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 1 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x4_L1e-3_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_W0" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-3_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_W0" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-3_W0.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-4_W4e-5" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 0 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x1_L1e-4_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-4_W4e-5" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 1 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x4_L1e-4_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-4_W5e-4" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-4_W5e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-4_W5e-4" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-4_W5e-4.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-4_W1e-4" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 0 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x1_L1e-4_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-4_W1e-4" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 1 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x4_L1e-4_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-4_W0" --n_optim_step 100 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-4_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-4_W0" --n_optim_step 100 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-4_W0.txt";

# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W1e-4" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 0 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W1e-4" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 1 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W5e-4" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W5e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W5e-4" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W5e-4.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W0" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 0 --weight_decay 0 > "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W0" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 1 --weight_decay 0 > "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W4e-5" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 2 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W4e-5" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 3 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W4e-5.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-2_S30_W1e-4" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 0 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x1_L1e-2_S30_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-2_S30_W1e-4" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 1 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x4_L1e-2_S30_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-2_S30_W5e-4" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-2_S30_W5e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-2_S30_W5e-4" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-2_S30_W5e-4.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-2_S30_W0" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 0 --weight_decay 0 > "ResNet34_Pretrained_CUB_x1_L1e-2_S30_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-2_S30_W0" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 1 --weight_decay 0 > "ResNet34_Pretrained_CUB_x4_L1e-2_S30_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-2_S30_W4e-5" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 2 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x1_L1e-2_S30_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-2_S30_W4e-5" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-2 --cuda 3 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x4_L1e-2_S30_W4e-5.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_S30_W1e-4" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 0 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x1_L1e-3_S30_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_S30_W1e-4" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 1 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x4_L1e-3_S30_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_S30_W5e-4" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-3_S30_W5e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_S30_W5e-4" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-3_S30_W5e-4.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_S30_W0" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 0 --weight_decay 0 > "ResNet34_Pretrained_CUB_x1_L1e-3_S30_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_S30_W0" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 1 --weight_decay 0 > "ResNet34_Pretrained_CUB_x4_L1e-3_S30_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_S30_W4e-5" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 2 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x1_L1e-3_S30_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_S30_W4e-5" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 3 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x4_L1e-3_S30_W4e-5.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-4_S30_W1e-4" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 0 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x1_L1e-4_S30_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-4_S30_W1e-4" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 1 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x4_L1e-4_S30_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-4_S30_W5e-4" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-4_S30_W5e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-4_S30_W5e-4" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-4_S30_W5e-4.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-4_S30_W0" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 0 --weight_decay 0 > "ResNet34_Pretrained_CUB_x1_L1e-4_S30_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-4_S30_W0" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 1 --weight_decay 0 > "ResNet34_Pretrained_CUB_x4_L1e-4_S30_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-4_S30_W4e-5" --n_optim_step 30 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 2 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x1_L1e-4_S30_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-4_S30_W4e-5" --n_optim_step 30 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-4 --cuda 3 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x4_L1e-4_S30_W4e-5.txt";

# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_S60_W1e-4" --n_optim_step 60 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 0 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x1_L1e-3_S60_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_S60_W1e-4" --n_optim_step 60 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 1 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x4_L1e-3_S60_W1e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_S60_W5e-4" --n_optim_step 60 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-3_S60_W5e-4.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_S60_W5e-4" --n_optim_step 60 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-3_S60_W5e-4.txt";


# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_S60_W0" --n_optim_step 60 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 0 --weight_decay 0 > "ResNet34_Pretrained_CUB_x1_L1e-3_S60_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_S60_W0" --n_optim_step 60 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 1 --weight_decay 0 > "ResNet34_Pretrained_CUB_x4_L1e-3_S60_W0.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-3_S60_W4e-5" --n_optim_step 60 --down_scale 1 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 2 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x1_L1e-3_S60_W4e-5.txt" \
# |
# python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-3_S60_W4e-5" --n_optim_step 60 --down_scale 4 \
#     --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-3 --cuda 3 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x4_L1e-3_S60_W4e-5.txt";


python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W1e-4" --n_optim_step 30 --down_scale 1 \
    --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 0 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W1e-4.txt" \
|
python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W1e-4" --n_optim_step 30 --down_scale 4 \
    --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 1 --weight_decay 1e-4 > "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W1e-4.txt" \
|
python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W5e-4" --n_optim_step 30 --down_scale 1 \
    --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 2 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W5e-4.txt" \
|
python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W5e-4" --n_optim_step 30 --down_scale 4 \
    --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 3 --weight_decay 5e-4 > "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W5e-4.txt";


python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W0" --n_optim_step 30 --down_scale 1 \
    --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 0 --weight_decay 0 > "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W0.txt" \
|
python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W0" --n_optim_step 30 --down_scale 4 \
    --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 1 --weight_decay 0 > "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W0.txt" \
|
python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W4e-5" --n_optim_step 30 --down_scale 1 \
    --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 2 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x1_L1e-1_S30_W4e-5.txt" \
|
python3 pretrainer_CUB.py --expr_name "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W4e-5" --n_optim_step 30 --down_scale 4 \
    --batch_size 128 --epochs 300 --pretrained --model_name resnet34 --learning_rate 1e-1 --cuda 3 --weight_decay 4e-5 > "ResNet34_Pretrained_CUB_x4_L1e-1_S30_W4e-5.txt";