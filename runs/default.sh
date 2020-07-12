export CUDA_VISIBLE_DEVICES=1
python train.py --model_name unet --exp_name loss_ce --loss ce --dataset_name DRIVE
python train.py --model_name unet --exp_name loss_tversky --loss tversky --dataset_name DRIVE

python train.py --model_name enet --exp_name w_ce --loss weighted_ce
python train.py --model_name enet --exp_name b_focal --loss b_focal ~!!
python train.py --model_name enet --exp_name c_focal --loss c_focal ~!!
python train.py --model_name enet --exp_name dice --loss dice ~!!



export CUDA_VISIBLE_DEVICES=1
python train.py --model_name unet_xception_resnetblock

