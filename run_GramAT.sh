
python3 Gram_AT.py --n_gpu 0 --num_workers 8 --batch_size 16 --LR_scale 4 --alpha 1.0 --beta 0.5 --pooling avg --use_grammian > ./KD_AT_logs/LR4_GRAM_AVG_A1_T4_B05_C1234.txt ;

# [LRScale]_[Attention]_[Pooling]_[Alpha]_[Temperature]_[Beta]_[Target]