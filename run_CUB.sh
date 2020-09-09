python3 pretrainer_CUB.py --down_scale 4 --epochs 300 --model_name resnet34 --learning_rate 1e-4 --pretrained --cuda 0 > ResNet34_CUB_x4_1E-4.txt \
|
python3 pretrainer_CUB.py --down_scale 1 --epochs 300 --model_name resnet34 --learning_rate 1e-4 --pretrained --cuda 1 > ResNet34_CUB_x1_1E-4.txt;

python3 pretrainer_CUB.py --down_scale 4 --epochs 300 --model_name resnet50 --learning_rate 1e-4 --pretrained --cuda 0> ResNet50_CUB_x4_1E-4.txt \
|
python3 pretrainer_CUB.py --down_scale 1 --epochs 300 --model_name resnet50 --learning_rate 1e-4 --pretrained --cuda 1> ResNet50_CUB_x1_1E-4.txt;
