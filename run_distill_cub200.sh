python3 CUB200_DIST_REFACTOR.py --expr_name "CUB200_A09_T2_B128_L1E-5" --alpha 0.9 --temperature 2 --gpus 0,1 --teacher_weight ./pretrained/CUB200/CUB200_ResNet34_HR.pth \
    --root ../dataset/ --batch_size 128 > CUB200_A09_T2_B64_L1E-5.txt \
|
python3 CUB200_DIST_REFACTOR.py --expr_name "CUB200_A05_T2_B128_L1E-5" --alpha 0.5 --temperature 2 --gpus 2,3 --teacher_weight ./pretrained/CUB200/CUB200_ResNet34_HR.pth \
    --root ../dataset/ --batch_size 128 --learning_rate 1e-5 > CUB200_A05_T2_B64_L1E-5.txt;