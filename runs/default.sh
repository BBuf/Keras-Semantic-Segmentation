export CUDA_VISIBLE_DEVICES=0
python train.py --model_name vggunet 
python train.py --model_name fcn8 

export CUDA_VISIBLE_DEVICES=1
python train.py --model_name unet_xception_resnetblock

