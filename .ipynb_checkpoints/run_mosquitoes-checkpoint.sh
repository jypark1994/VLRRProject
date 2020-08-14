# echo "Training x1 models"
# CUDA_VISIBLE_DEVICES=0 python3 pretrainer.py --dataset mosquitoes --down_scale 1 --model_name resnet18 --batch_size 64 --num_workers 8 --pretrained > ./models/mosquitoes/resnet18_x1_pt.txt \
# |
# CUDA_VISIBLE_DEVICES=1 python3 pretrainer.py --dataset mosquitoes --down_scale 1 --model_name resnet34 --batch_size 64 --num_workers 8 --pretrained > ./models/mosquitoes/resnet34_x1_pt.txt \
# |
# CUDA_VISIBLE_DEVICES=2,3 python3 pretrainer.py --dataset mosquitoes --down_scale 1 --model_name resnet50 --batch_size 64 --num_workers 8 --pretrained --multi_gpu > ./models/mosquitoes/resnet50_x1_pt.txt;

echo "Training x2 models"
CUDA_VISIBLE_DEVICES=0 python3 pretrainer.py --dataset mosquitoes --down_scale 2 --model_name resnet18 --batch_size 64 --num_workers 8 --pretrained > ./models/mosquitoes/resnet18_x2_pt.txt \
|
CUDA_VISIBLE_DEVICES=1 python3 pretrainer.py --dataset mosquitoes --down_scale 2 --model_name resnet34 --batch_size 64 --num_workers 8 --pretrained > ./models/mosquitoes/resnet34_x2_pt.txt \
|
CUDA_VISIBLE_DEVICES=2,3 python3 pretrainer.py --dataset mosquitoes --down_scale 2 --model_name resnet50 --batch_size 64 --num_workers 8 --pretrained --multi_gpu > ./models/mosquitoes/resnet50_x2_pt.txt;

echo "Training x4 models"
CUDA_VISIBLE_DEVICES=0 python3 pretrainer.py --dataset mosquitoes --down_scale 4 --model_name resnet18 --batch_size 64 --num_workers 8 --pretrained > ./models/mosquitoes/resnet18_x4_pt.txt \
|
CUDA_VISIBLE_DEVICES=1 python3 pretrainer.py --dataset mosquitoes --down_scale 4 --model_name resnet34 --batch_size 64 --num_workers 8 --pretrained > ./models/mosquitoes/resnet34_x4_pt.txt \
|
CUDA_VISIBLE_DEVICES=2,3 python3 pretrainer.py --dataset mosquitoes --down_scale 4 --model_name resnet50 --batch_size 64 --num_workers 8 --pretrained --multi_gpu > ./models/mosquitoes/resnet50_x4_pt.txt;

echo "Training x8 models"
CUDA_VISIBLE_DEVICES=0 python3 pretrainer.py --dataset mosquitoes --down_scale 8 --model_name resnet18 --batch_size 64 --num_workers 8 --pretrained > ./models/mosquitoes/resnet18_x8_pt.txt \
|
CUDA_VISIBLE_DEVICES=1 python3 pretrainer.py --dataset mosquitoes --down_scale 8 --model_name resnet34 --batch_size 64 --num_workers 8 --pretrained > ./models/mosquitoes/resnet34_x8_pt.txt \
|
CUDA_VISIBLE_DEVICES=2,3 python3 pretrainer.py --dataset mosquitoes --down_scale 8 --model_name resnet50 --batch_size 64 --num_workers 8 --pretrained --multi_gpu > ./models/mosquitoes/resnet50_x8_pt.txt;

echo "Training x16 models"
CUDA_VISIBLE_DEVICES=0 python3 pretrainer.py --dataset mosquitoes --down_scale 16 --model_name resnet18 --batch_size 64 --num_workers 8 --pretrained > ./models/mosquitoes/resnet18_x16_pt.txt \
|
CUDA_VISIBLE_DEVICES=1 python3 pretrainer.py --dataset mosquitoes --down_scale 16 --model_name resnet34 --batch_size 64 --num_workers 8 --pretrained > ./models/mosquitoes/resnet34_x16_pt.txt \
|
CUDA_VISIBLE_DEVICES=2,3 python3 pretrainer.py --dataset mosquitoes --down_scale 16 --model_name resnet50 --batch_size 64 --num_workers 8 --pretrained --multi_gpu > ./models/mosquitoes/resnet50_x16_pt.txt;

