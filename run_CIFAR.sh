CUDA_VISIBLE_DEVICES=0 python3 pretrainer_CIFAR.py --down_scale 1 --weight_decay 1e-4 --interpolate \
    --model_name resnet32 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet34_w1e-4_x1_pt.txt \
|
CUDA_VISIBLE_DEVICES=1 python3 pretrainer_CIFAR.py --down_scale 4 --weight_decay 1e-4 --interpolate \
    --model_name resnet32 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet34_w1e-4_x4_pt.txt \
|
CUDA_VISIBLE_DEVICES=2 python3 pretrainer_CIFAR.py --down_scale 1 --weight_decay 5e-4 --interpolate \
    --model_name resnet32 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet34_w5e-4_x1_pt.txt \
|
CUDA_VISIBLE_DEVICES=3 python3 pretrainer_CIFAR.py --down_scale 4 --weight_decay 5e-4 --interpolate \
    --model_name resnet32 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet34_w5e-4_x4_pt.txt;

CUDA_VISIBLE_DEVICES=0 python3 pretrainer_CIFAR.py --down_scale 1 --weight_decay 4e-5 --interpolate \
    --model_name resnet32 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet34_w4e-5_x1_pt.txt \
|
CUDA_VISIBLE_DEVICES=1 python3 pretrainer_CIFAR.py --down_scale 4 --weight_decay 4e-5 --interpolate \
    --model_name resnet32 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet34_w4e-5_x4_pt.txt \
|
CUDA_VISIBLE_DEVICES=2 python3 pretrainer_CIFAR.py --down_scale 1 --weight_decay 0 --interpolate \
    --model_name resnet32 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet34_w0_x1_pt.txt \
|
CUDA_VISIBLE_DEVICES=3 python3 pretrainer_CIFAR.py --down_scale 4 --weight_decay 0 --interpolate \
    --model_name resnet32 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet34_w0_x4_pt.txt;