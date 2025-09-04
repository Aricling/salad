## 启动在MotionBert数据上训练vae的代码,save_latest可以理解成多少step保存一次
python train_vae.py --name train_vae_on_MB_rep_try_1 --dataset_name t2m --gpu_id 1 --save_latest 5000

## 启动salad中diffusion训练的代码
CUDA_VISIBLE_DEVICES=5 python train_denoiser.py --name train_wi_fted_clip_try_0 --vae_name t2m_vae_gelu