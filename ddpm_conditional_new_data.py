
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from utils import _structural_similarity_index, _peak_signal_noise_ratio, _mean_squared_error
from modules import UNet_conditional, EMA, UNet_conditional_concat, UNet_conditional_fully_concat, UNet_conditional_fully_add 
from modules import UNet_conditional_concat_with_mask, UNet_conditional_concat_with_mask_v2, UNet_conditional_concat_Large
from modules import UNet_conditional_concat_XLarge, UNet_conditional_concat_pseudo_3D
import logging
from torch.utils.tensorboard import SummaryWriter
# from inpainting.challenge_metrics_2023 import generate_metrics

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
"""
需要修改的地方：
7:加上实际的评价指标
15:使用相邻slice做guidance
16:sample之后重新变成图片这一步可能有问题，不是直接归一化，可以用其他操作
17:sample之后加一个inpaint，输出图像
20:同一个人用相同的schedule去燥，好像不太行？
21:层间不一致的问题
22:边界添加，拼接
24:添加自动生成结果
25:自动计算量化结果
26:修改loss，将PSNR 作为loss放进去。平衡全局loss和局部的权重。
27:添加transformer的代码
28:尝试不同的UNET，使用UNET+ 或者UNET++
29:加一些即插即用的模块，例如SENET，GAM
30:看下不同步长的效果
31:告诉模型生的区域是那块区域，输入这个mask的坐标，希望他能学到这个区域大概的样子
32:试一下group norm
33:试一下打乱效果会不会好一些
34:试一下用视频生成的方式来生成MRI，看下效果会不会好点
35:Masked Diffusion model:https://arxiv.org/pdf/2308.05695
36:可以试一下加各种编码
37:Mamba复现 UMamba Swin-UMamba ... https://www.bilibili.com/read/cv31896604/
38:完成128和192尺寸的UNET
39:实现添加位置信息的UNET
40:加一段sample合成一个nifti的代码
41:复现i2sb和rddpm的代码,用残差生成
42:加上数据增强的代码，旋转，滑窗
43:可以用多级的信息来一起生成
"""

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=240, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def _timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        # model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x

def inpaint_image(original_image, generated_image, mask):
    """
    将生成的图像融合到原始图像的指定区域中。
    
    参数:
    original_image (torch.Tensor): 原始图像
    generated_image (torch.Tensor): 生成的图像
    mask (torch.Tensor): 掩码图像, 1表示需要inpaint的区域, 0表示保留原图
    
    返回:
    torch.Tensor: 输出的合成图像
    """
    # 将三个输入tensor转换到相同的设备上
    device = original_image.device
    mask = mask.to(device)
    generated_image = generated_image.to(device)
    
    # 使用掩码融合原图和生成的图像
    output_image = original_image.clone()
    # print(output_image.shape, mask.shape, generated_image.shape)
    output_image = output_image * (1 - mask) + generated_image * mask
    
    return output_image


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data_new(args)
    # test_dataloader = get_test_data(args)
    # model = UNet_conditional().to(device)
    model = UNet_conditional_concat_XLarge().to(device)
    # model = UNet_conditional_fully_concat().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.999)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, cropped_images, masks, image_without_healthy) in enumerate(pbar):
            # print(images.shape)
            # images = images.to(device)
            labels = cropped_images
            b, c, l, w = images.shape

            # print(slice) #size 应该是 b, 1, 240, 240
            images_slice = images[:,:,:,:]
            labels_slice = labels[:,:,:,:]
            masks_slice = masks[:,:,:,:]
            image_without_healthy = image_without_healthy[:,:,:,:]

            images_slice = images_slice.to(device)
            labels_slice = labels_slice.to(device)
            masks_slice = masks_slice.to(device)
            image_without_healthy = image_without_healthy.to(device)

            images_slice = images_slice.to(torch.float)
            labels_slice = labels_slice.to(torch.float)
            masks_slice = masks_slice.to(torch.float)
            image_without_healthy_slice = image_without_healthy.to(torch.float)
            # print('input shape', images_slice.shape)


            t = diffusion._timesteps(images_slice.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images_slice, t)
            predicted_noise = model(x_t, t, image_without_healthy_slice, masks_slice)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # if epoch % 5 == 0:
            # with torch.no_grad():
            #     pbar_test = tqdm(test_dataloader)
            #     ssim_list = []
            #     psnr_list = []
            #     mse_list = []
            #     for i, (images, cropped_images, masks) in enumerate(pbar_test):
            #         modefied_images = cropped_images
            #         b, _, _, _ = images.shape
            #         d_images = diffusion.sample(model, n=b, labels=modefied_images, masks=masks)
            #         images_predict_slice, _, _ = inpaint_image(images[:,:,:,:], d_images[:,:,:,:], masks[:,:,:,:])
            #         img = image_preprocess_tensor(img)
            #         d_img = image_preprocess(d_img)

            #         img_copy = img.unsqueeze(0).unsqueeze(0)
            #         d_img_copy = d_img.unsqueeze(0).unsqueeze(0)
            #         masks = masks.cpu()
            #         mask_copy = masks.unsqueeze(0).unsqueeze(0)

            #         mask_copy = mask_copy.bool()
            #         images = images.cpu()
            #         d_images_new = images_predict_slice.cpu()
            #         dd = d_images[:,:,:,:].cpu()
            #         cropped_images = modefied_images.cpu()

            #         for index in range(args.batch_size):
            #             img = images[index, 0, :, :]
            #             d_img = d_images_new[index, 0, :, :]
            #             c_img = cropped_images[index, 0, :, :]
            #             dd_img = dd[index, 0, :, :]

            #             ssim = _structural_similarity_index(
            #                     target=img_copy,
            #                     prediction=d_img_copy,
            #                     mask=mask_copy,
            #                 ).item()
            #             mse = _mean_squared_error(
            #                     target=img_copy,
            #                     prediction=d_img_copy,
            #                     squared=True,
            #                 ).item()
            #             psnr = _peak_signal_noise_ratio(
            #                     target=img_copy,
            #                     prediction=d_img_copy,
            #                 ).item()
                        
            #             ssim_list.append(ssim)
            #             psnr_list.append(psnr)
            #             mse_list.append(mse)

            #     # save_images(d_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            #     save_images(img, c_img, dd_img, d_img, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
    


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 500
    args.batch_size = 2
    args.image_size = 256
    # args.dataset_path =  r"D:\ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
    args.dataset_path =  r"C:\Users\DELL\Desktop\DDPM\ddpm_brats\DDPM_brain\test_data"
    args.device = "cuda"
    args.lr = 1e-4
    args.train = True
    args.shuffle = True
    args.random_seed = 2024
    train(args)


if __name__ == '__main__':
    launch()


