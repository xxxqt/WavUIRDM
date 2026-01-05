import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
from  src.recon import DRM
from src.degrad_classify import Diff_NoImg_DC, Diff_DC
import sys
import argparse
from data.universal_dataset import AlignedDataset_all
from src.model import (ResidualDiffusion,Trainer, Unet, UnetRes,set_seed)

def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='/home/data/')
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', type=bool, default=True, help='if specified, do not flip the images for data augmentation')
    parser.add_argument("--bsize", type=int, default=2)
    opt = parser.parse_args()
    return opt

sys.stdout.flush()
set_seed(10)

save_and_sample_every = 1000
if len(sys.argv) > 1:
    sampling_timesteps = int(sys.argv[1])
else:
    sampling_timesteps = 5
train_num_steps = 100000

condition = True

train_batch_size = 1
num_samples = 1
image_size = 256
opt = parsr_args()
results_folder = "./ckpt_universal/diffuir"
dataset = AlignedDataset_all(opt, image_size, augment_flip=False, equalizeHist=True, crop_patch=False, generation=False,task='blur')
num_unet = 1
objective = 'pred_res'
test_res_or_noise = "res"
sampling_timesteps = 3
sum_scale = 0.01
ddim_sampling_eta = 0.
delta_end = 1.8e-3

HFRM_model = DRM(
    in_channel=3,
    dim=24,
    mid_blk_num=3,
    enc_blk_nums=[1, 1, 1, 1],
    dec_blk_nums=[1, 1, 1, 1]
)

cls_model = Diff_DC(
    feature_dims=[24, 48, 96, 192],
    num_res_blocks=2,
    num_classes=5
)


model = UnetRes(
    dim=32,
    dim_mults=(1, 2, 2, 4),
    num_unet=num_unet,
    condition=condition,
    objective=objective,
    test_res_or_noise = test_res_or_noise,
wave_condition = True
)

diffusion = ResidualDiffusion(
    model,
    refine_model=HFRM_model ,
    cls_model=cls_model,
    refine_ckpt_path=None,
    image_size=image_size,
    timesteps=1000,
    delta_end = delta_end,
    sampling_timesteps=sampling_timesteps,
    objective=objective,
    loss_type='l1',
    condition=condition,
    sum_scale=sum_scale,
    test_res_or_noise = test_res_or_noise,
    wave_cond=True
)

trainer = Trainer(
    diffusion,
    dataset,
    opt,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=8e-5,
    train_num_steps=train_num_steps,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=False,
    results_folder = results_folder,
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    num_unet=num_unet,
)

# test
if not trainer.accelerator.is_local_main_process:
    pass
else:
    trainer.load(30)
    trainer.set_results_folder('./result')
    trainer.test(last=True)

