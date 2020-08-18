
python3 Gram_AT.py --n_gpu 0 --num_workers 8 --batch_size 16 --LR_scale 4 --alpha 1 --beta 0.5 --temperature 10 --pooling avg \
 > ./KD_AT_logs/LR4_MSE_AVG_A1_T4_B03_C1234.txt ;

python3 Gram_AT.py --n_gpu 0 --num_workers 8 --batch_size 16 --LR_scale 4 --alpha 1 --beta 0.5 --temperature 20 --pooling avg \
 > ./KD_AT_logs/LR4_MSE_AVG_A1_T4_B08_C1234.txt ;

# [LRScale]_[Attention]_[Pooling]_[Alpha]_[Temperature]_[Beta]_[Target]