python3 CUB200_DIST_REFACTOR.py --expr_name "CUB200_A07_T5_B256_L1E-4" --alpha 0.7 --temperature 5 --gpus 0,1,2,3 \
    --teacher_weight ./pretrained/CUB200/CUB200_ResNet34_HR.pth \
    --root ../dataset/ --batch_size 256 --learning_rate 1e-4 > CUB200_A07_T5_B256_L1E-4.txt;

python3 CUB200_DIST_REFACTOR.py --expr_name "CUB200_A09_T5_B256_L1E-4" --alpha 0.9 --temperature 5 --gpus 0,1,2,3 \
    --teacher_weight ./pretrained/CUB200/CUB200_ResNet34_HR.pth \
    --root ../dataset/ --batch_size 256 --learning_rate 1e-4 > CUB200_A09_T5_B256_L1E-4.txt;