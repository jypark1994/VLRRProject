python3 pretrainer_CUB.py --expr_name "ResNet34_CUB_x4_1E-5" --down_scale 4 --batch_size 128 --epochs 300 --model_name resnet34 --learning_rate 1e-5 --pretrained --cuda 0,1 \
|
python3 pretrainer_CUB.py --expr_name "ResNet34_CUB_x1_1E-5" --down_scale 1 --batch_size 128 --epochs 300 --model_name resnet34 --learning_rate 1e-5 --pretrained --cuda 2,3 ;

python3 pretrainer_CUB.py --expr_name "ResNet50_CUB_x4_1E-5" --down_scale 4 --batch_size 128 --epochs 300 --model_name resnet50 --learning_rate 1e-5 --pretrained --cuda 0,1 \
|
python3 pretrainer_CUB.py --expr_name "ResNet50_CUB_x1_1E-5" --down_scale 1 --batch_size 128 --epochs 300 --model_name resnet50 --learning_rate 1e-5 --pretrained --cuda 2,3 ;
