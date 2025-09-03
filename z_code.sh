## 启动在MotionBert数据上训练vae的代码
python train_vae.py --name train_vae_on_MB_rep_try_0 --dataset_name t2m --gpu_id 2

## 启动salad中diffusion训练的代码
CUDA_VISIBLE_DEVICES=5 python train_denoiser.py --name train_wi_fted_clip_try_0 --vae_name t2m_vae_gelu