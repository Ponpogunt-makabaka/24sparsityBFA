DATE=$(date +%Y-%m-%d)
mkdir -p ./save/${DATE}/

python3 zhwf1_main_nm.py --dataset cifar10 --data_path /home/lab-2010/Documents/zhwf/BFA/data \
  --arch quan_resnet20_nm --save_path ./save/quan_resnet20_nm_BFA \
  --test_batch_size 128 --workers 8 --ngpu 1 --gpu_id 1 --print_freq 50 \
  --evaluate --resume pth/1_sparse_finetune_INT8.pth --fine_tune --reset_weight --bfa \
  --n_iter 1 --attack_sample_size 128
