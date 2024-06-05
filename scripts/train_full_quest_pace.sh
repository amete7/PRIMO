task_name=${1}
exp_name=${2}
seed=${3}
# config_name=${alg_name}
# exp_name=${3}
# group=${4}
# wandb_name=${task_name}-${alg_name}-${exp_name}
# run_dir="data/outputs/${task_name}/${alg_name}/${addition_info}/seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
# gpu_id=0
# echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


# if [ $DEBUG = True ]; then
#     wandb_mode=offline
#     # wandb_mode=online
#     echo -e "\033[33mDebug mode!\033[0m"
#     echo -e "\033[33mDebug mode!\033[0m"
#     echo -e "\033[33mDebug mode!\033[0m"
# else
#     wandb_mode=online
#     echo -e "\033[33mTrain mode\033[0m"
# fi


# export HYDRA_FULL_ERROR=1 
# export CUDA_VISIBLE_DEVICES=${gpu_id}
# python train.py --config-name=${config_name}.yaml \
#                             task=${task_name} \
#                             hydra.run.dir=${run_dir} \
#                             training.debug=$DEBUG \
#                             training.seed=${seed} \
#                             training.device="cuda:0" \
#                             exp_name=${exp_name} \
#                             logging.mode=${wandb_mode} \
#                             logging.group=${group} \
#                             logging.name=wandb_name \
#                             checkpoint.save_ckpt=${save_ckpt} \
#                             ${@:6}

python train.py --config-name=train_autoencoder.yaml \
    task=${task_name} \
    exp_name=${exp_name} \
    seed=${seed} \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    make_unique_experiment_dir=false

python train.py --config-name=train_prior.yaml \
    task=${task_name} \
    exp_name=${exp_name} \
    seed=${seed} \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.auto_continue=true \
    train_dataloader.persistent_workers=true \
    make_unique_experiment_dir=false

                                