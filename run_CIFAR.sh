CUDA_VISIBLE_DEVICES=0 python3 pretrainer_CIFAR.py --down_scale 1 --interpolate \
    --model_name resnet32 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet34_x1_pt.txt \
|
CUDA_VISIBLE_DEVICES=1 python3 pretrainer_CIFAR.py --down_scale 4 --interpolate \
    --model_name resnet32 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet34_x4_pt.txt;