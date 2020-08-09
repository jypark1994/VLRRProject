echo "Training x1 (32x32)"
python3 pretrainer_CIFAR.py --down_scale 1 --interpolate --model_name resnet18 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet18_x1_pt.txt;

echo "Training x2 (16x16)"
python3 pretrainer_CIFAR.py --down_scale 2 --interpolate --model_name resnet18 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet18_x2_pt.txt;

echo "Training x4 (8x8)"
python3 pretrainer_CIFAR.py --down_scale 4 --interpolate --model_name resnet18 --batch_size 128 --num_workers 8 --pretrained > ./models/CIFAR10/resnet18_x4_pt.txt;