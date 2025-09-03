import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader
import numpy as np

from options.vae_option import arg_parse
from models.vae.model import VAE
from models.vae.trainer import VAETrainer
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from data.t2m_dataset import MotionDataset

from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.fixseed import fixseed

from MotionBERT.infer_wild_multiple import plot_MB_rep_interface

os.environ["OMP_NUM_THREADS"] = "1"

def plot_t2m(MB_motion_data = None, name_list = None, texts_list = None, save_dir = None):
    # data = train_dataset.inv_transform(data)
    motion_emb_mean = np.load("/home/mengqing/usr/motion-diffusion-model/dataset/motion_emb_mean.npy")
    motion_emb_std = np.load("/home/mengqing/usr/motion-diffusion-model/dataset/motion_emb_std.npy")
    for i in range(len(MB_motion_data)):
        MB_motion = MB_motion_data[i:i+1]
        name = [name_list[i%4]]
        if i // 4 ==0:
            save_path = pjoin(save_dir, 'gt')
        else:
            save_path = pjoin(save_dir, 'pred')
        plot_MB_rep_interface(MB_motion, seq_names = name, output_path = save_path,
                              mean=motion_emb_mean, std=motion_emb_std)



if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)
    fixseed(opt.seed)

    # model
    net = VAE(opt)
    num_params = sum(param.numel() for param in net.parameters())
    print('Total trainable parameters of all models: {}M'.format(num_params/1_000_000))

    # evaluation setup
    wrapper_opt = get_opt(opt.dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_val_loader, _ = get_dataset_motion_loader(opt.dataset_opt_path, 32, 'val', device=opt.device)  ## 这个用的其实就是val.txt

    # dataset & dataloader
    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy')) ## ./checkpoints/t2m/Comp_v6_KLD005/meta
    std = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')    ## ./dataset/humanml3d/
    val_split_file = pjoin(opt.data_root, 'val.txt')   ## LOOK UP! 其实被我改成了test，不是eval了

    train_dataset = MotionDataset(opt, mean, std, train_split_file)
    val_dataset = MotionDataset(opt, mean, std, val_split_file) ## 这个split_file其实和上面的eval_val_loader

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers, shuffle=True, pin_memory=True) ## bs=256
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers, shuffle=True, pin_memory=True)

    # train
    trainer = VAETrainer(opt, net)
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, plot_t2m)

## python train_vae.py --name vae_silu_kl2e-2_noposloss --vae_type vae --lambda_kl 2e-2 --activation silu