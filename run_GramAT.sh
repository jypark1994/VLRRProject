
python3 Gram_AT.py --n_gpu 0 --num_workers 4 --batch_size 32 --LR_scale 4 --beta 0.5 > ./KD_AT_logs/LR4_MSE_A09_T4_B05_C1234.txt \
|
python3 Gram_AT.py --n_gpu 0 --num_workers 4 --batch_size 32 --LR_scale 4 > ./KD_AT_logs/LR4_MSE_A09_T4_B01_C1234.txt ;

python3 Gram_AT.py --n_gpu 0 --num_workers 4 --batch_size 16 --LR_scale 2 --beta 0.5 > ./KD_AT_logs/LR2_MSE_A09_T4_B05_C1234.txt \
|
python3 Gram_AT.py --n_gpu 0 --num_workers 4 --batch_size 16 --LR_scale 2 > ./KD_AT_logs/LR2_MSE_A09_T4_B01_C1234.txt ;

# python3 Gram_AT.py --n_gpu 0 --num_workers 4 --LR_scale 4 --use_grammian > ./KD_AT_logs/LR4_GRAM_A09_T4_B01_C1234.txt \
# |
# python3 Gram_AT.py --n_gpu 0 --num_workers 4 --LR_scale 4 --use_grammian --beta 0.5 > ./KD_AT_logs/LR4_GRAM_A09_T4_B05_C1234.txt;



# [LRScale]_[Attention]_[Alpha]_[Temperature]_[Beta]_[Target]