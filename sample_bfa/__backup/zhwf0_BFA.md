DATE=$(date +%Y-%m-%d)
python3 main.py --dataset cifar10 --data_path /home/lab-2010/Documents/zhwf/BFA/data \
  --arch resnet20_quan --save_path ./save/cifar10_resnet20_quan_BFA_Baseline \
  --test_batch_size 128 --workers 8 --ngpu 1 --gpu_id 1 --print_freq 50 \
  --evaluate --resume pth/0_model_best.pth.tar --fine_tune --reset_weight --bfa \
  --n_iter 20 --attack_sample_size 128





python3 scripts/analyze_attack_profile.py <path/to/attack_profile.csv>
python3 scripts/detect_flipped_bits.py <attack_profile.csv> [--n_bits N]
