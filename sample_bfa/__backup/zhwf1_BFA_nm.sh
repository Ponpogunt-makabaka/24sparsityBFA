

data_path='./data'

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=resnet20
dataset=cifar10
test_batch_size=128

label_info=BFA_NM

attack_sample_size=128 # number of data used for BFA
n_iter=20 # number of iteration to perform BFA
k_top=10 # only check k_top weights with top gradient ranking in each layer

save_path=./save/${DATE}/${dataset}_${model}_${label_info}

# set the pretrained model path
pretrained_model=/home/lab-2010/Documents/zhwf/BFA/pth/1_sparse_finetune.pth

############### Neural network ############################
COUNTER=0
{
while [ $COUNTER -lt 1 ]; do
    $PYTHON zhwf1_main_nm.py --dataset ${dataset} \
        --data_path ${data_path}   \
        --arch ${model} --save_path ${save_path}  \
        --test_batch_size ${test_batch_size} --workers 8 --ngpu 1 --gpu_id 1 \
        --print_freq 50 \
        --evaluate --resume ${pretrained_model} --fine_tune\
        --reset_weight --bfa --n_iter ${n_iter} \
        --attack_sample_size ${attack_sample_size} \

    let COUNTER=COUNTER+1
done
} &

wait