from PIL import Image
from matplotlib import pyplot as plt
import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import glob
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
)
# from inpainting.challenge_metrics_2023 import generate_metrics, read_nifti_to_tensor

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([image for image in images.cpu()], dim=-1)
        ], dim=-2).permute(1,2,0).cpu())
    plt.show()

def save_images(img, c_img, dd_img, d_img, path, **kwargs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    plt.figure(figsize=(8, 8))
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title('Ground-truth')
    axes[0,1].imshow(c_img, cmap='gray')
    axes[0,1].set_title('Cropped guidance')
    axes[1,0].imshow(dd_img, cmap='gray')
    axes[1,0].set_title('DDPM genarated')
    axes[1,1].imshow(d_img, cmap='gray')
    axes[1,1].set_title('Final infilled image')
    plt.savefig(path)

def image_preprocess(image):
    t1_clipped = np.clip(
                        image,
                        np.quantile(image, 0.001),
                        np.quantile(image, 0.999),
                    )
    t1_normalized = (t1_clipped - np.min(t1_clipped)) / (
        np.max(t1_clipped) - np.min(t1_clipped)
    )

    return t1_normalized

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, train=True, device='cuda'):
        self.data_dir = data_dir
        self.train = train
        self.device = device
        self.subject_dirs = sorted(glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True))
        train_size = int(0.9 * len(self.subject_dirs))
        if self.train:
            self.subject_dirs = self.subject_dirs[:train_size]
        else:
            self.subject_dirs = self.subject_dirs[train_size:]
        # print(self.subject_dirs)

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_slice_path = self.subject_dirs[idx]
        with np.load(subject_slice_path) as data:
            image = data['image']
            healthy_mask = data['healthy_mask']
            cropped_image = data['cropped_image']
            # unhealthy_mask = data['unhealthy_mask']
        # image 是健康的图像，即只抠掉肿瘤区域的图像 cropped_image是抠掉要生成区域的图像，mask是健康图像的掩膜
        image = torch.from_numpy(image).squeeze(-1)
        cropped_image = torch.from_numpy(cropped_image).squeeze(-1)
        healthy_mask = torch.from_numpy(healthy_mask).squeeze(-1)
        return image.to(self.device), cropped_image.to(self.device), healthy_mask.to(self.device)

class BrainTumorDataset_inference(Dataset):
    def __init__(self, data_dir, train=True, device='cuda'):
        self.data_dir = data_dir
        self.train = train
        self.device = device
        self.subject_dirs = sorted(glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True))
        train_size = int(0.9 * len(self.subject_dirs))
        if self.train:
            self.subject_dirs = self.subject_dirs[:]
        else:
            self.subject_dirs = self.subject_dirs[train_size:]
        # print(self.subject_dirs)

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_slice_path = self.subject_dirs[idx]
        with np.load(subject_slice_path) as data:
            image = data['image']
            healthy_mask = data['healthy_mask']
            cropped_image = data['cropped_image']
            # unhealthy_mask = data['unhealthy_mask']
        # image 是健康的图像，即只抠掉肿瘤区域的图像 cropped_image是抠掉要生成区域的图像，mask是健康图像的掩膜
        image = torch.from_numpy(image).squeeze(-1)
        cropped_image = torch.from_numpy(cropped_image).squeeze(-1)
        healthy_mask = torch.from_numpy(healthy_mask).squeeze(-1)
        return image.to(self.device), cropped_image.to(self.device), healthy_mask.to(self.device)
    
class BrainTumorDataset_new(Dataset):
    def __init__(self, data_dir, train=True, device='cuda'):
        self.data_dir = data_dir
        self.train = train
        self.device = device
        self.subject_dirs = sorted(glob.glob(os.path.join(data_dir, '**', '256*.npz'), recursive=True))
        train_size = int(0.9 * len(self.subject_dirs))
        if self.train:
            self.subject_dirs = self.subject_dirs[:train_size]
        else:
            self.subject_dirs = self.subject_dirs[train_size:]
        # print(self.subject_dirs)

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_slice_path = self.subject_dirs[idx]
        with np.load(subject_slice_path) as data:
            image = data['image']
            healthy_mask = data['healthy_mask']
            cropped_image = data['cropped_image']
            cropped_image_preinfilled=data['cropped_image_preinfilled']
            # geometric_list=data['geometric_list']
            # adjacency_image=data['adjacency_image']
            image_without_healthy_area=data['image_without_healthy_area']

            # unhealthy_mask = data['unhealthy_mask']
        # image 是健康的图像，即只抠掉肿瘤区域的图像 cropped_image是抠掉要生成区域的图像，mask是健康图像的掩膜
        image = torch.from_numpy(image).squeeze(-1)
        cropped_image = torch.from_numpy(cropped_image).squeeze(-1)
        healthy_mask = torch.from_numpy(healthy_mask).squeeze(-1)
        cropped_image_preinfilled = torch.from_numpy(cropped_image_preinfilled).squeeze(-1)
        # adjacency_image = torch.from_numpy(adjacency_image).squeeze(-1)
        image_without_healthy_area = torch.from_numpy(image_without_healthy_area).squeeze(-1)
        return image.to(self.device), cropped_image.to(self.device), healthy_mask.to(self.device), image_without_healthy_area.to(self.device), cropped_image_preinfilled.to(self.device)

# 数据加载
# train_dataset = BrainTumorDataset('D:\\BraTS\\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')
# dataset = BrainTumorDataset('D:\\BraTS\\ASNR_test')
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

def get_data(args):
    dataset = BrainTumorDataset(args.dataset_path, args.train, device=args.device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    return dataloader

def get_data_inference(args):
    dataset = BrainTumorDataset_inference(args.dataset_path, args.train, device=args.device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    return dataloader

def get_test_data(args):
    dataset = BrainTumorDataset(args.dataset_path, train=False, device=args.device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def get_data_new(args):
    dataset = BrainTumorDataset_new(args.dataset_path, args.train, device=args.device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    return dataloader


def setup_logging(run_name):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{run_name}', exist_ok=True)
    os.makedirs(f'models/{run_name}', exist_ok=True)

def data_aug(image, mask):
    pass

def image_preprocess_tensor(tensor):
    # 首先将tensor转换为numpy数组
    image = tensor.numpy()
    
    # 对numpy数组进行clip和归一化操作
    t1_clipped = np.clip(image, np.quantile(image, 0.001), np.quantile(image, 0.999))
    t1_normalized = (t1_clipped - np.min(t1_clipped)) / (np.max(t1_clipped) - np.min(t1_clipped))
    
    # 将归一化后的numpy数组转换回tensor
    normalized_tensor = torch.from_numpy(t1_normalized)
    
    return normalized_tensor

def _structural_similarity_index(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Computes the structural similarity index between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
        mask (torch.Tensor, optional): The mask tensor. Defaults to None.

    Returns:
        torch.Tensor: The structural similarity index.
    """
    ssim = StructuralSimilarityIndexMeasure(return_full_image=True)
    _, ssim_idx_full_image = ssim(preds=prediction, target=target)
    mask = torch.ones_like(ssim_idx_full_image) if mask is None else mask
    # ssim_idx = None
    # mask = mask.bool()
    # print(mask.bool())
    # print(ssim_idx_full_image.shape)
    try:
        ssim_idx = ssim_idx_full_image[mask]
        # print('here')
    except Exception as e:
        print(f"Error: {e}")
        if len(ssim_idx_full_image.shape) == 0:
            ssim_idx = torch.ones_like(mask) * ssim_idx_full_image
    return ssim_idx.mean()


def _mean_squared_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    squared: bool = True,
) -> torch.Tensor:
    """
    Computes the mean squared error between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
        TODO update documentation
    """
    mse = MeanSquaredError(
        squared=squared,
    )

    return mse(preds=prediction, target=target)


def _peak_signal_noise_ratio(
    target: torch.Tensor,
    prediction: torch.Tensor,
    data_range: tuple = None,
    epsilon: float = None,
) -> torch.Tensor:
    """
    Computes the peak signal to noise ratio between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
        data_range (tuple, optional): If not None, this data range (min, max) is used as enumerator instead of computing it from the given data. Defaults to None.
        epsilon (float, optional): If not None, this epsilon is added to the denominator of the fraction to avoid infinity as output. Defaults to None.
    """

    if epsilon == None:
        psnr = (
            PeakSignalNoiseRatio()
            if data_range == None
            else PeakSignalNoiseRatio(data_range=data_range[1] - data_range[0])
        )
        return psnr(preds=prediction, target=target)
    else:  # implementation of PSNR that does not give 'inf'/'nan' when 'mse==0'
        mse = _mean_squared_error(target=target, prediction=prediction)
        if data_range == None:  # compute data_range like torchmetrics if not given
            min_v = (
                0 if torch.min(target) > 0 else torch.min(target)
            )  # look at this line
            max_v = torch.max(target)
        else:
            min_v, max_v = data_range
        return 10.0 * torch.log10(((max_v - min_v) ** 2) / (mse + epsilon))


def _mean_squared_log_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the mean squared log error between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
    """
    mle = MeanSquaredLogError()
    return mle(preds=prediction, target=target)


def _mean_absolute_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the mean absolute error between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
    """
    mae = MeanAbsoluteError()
    return mae(preds=prediction, target=target)