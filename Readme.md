CUDA_VISIBLE_DEVICES=7 \
python train.py --t_task 4 --n_way 5 --k_shot 1 --k_query 1 --num_workers 4 --use_dali t \
python train.py --t_task 2 --n_way 5 --k_shot 5 --k_query 15 --num_workers 4 --use_dali t --mem_size 10 --start_epoch 162