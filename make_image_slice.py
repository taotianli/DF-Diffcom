import os
import numpy as np
import torch
import glob
import nibabel as nib
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

data_dir = '/hpc/data/home/bme/yubw/taotl/BraTS-2023_challenge/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training/'
subject_dirs = sorted([os.path.join(data_dir, name) for name in os.listdir(data_dir)])
# print(subject_dirs)
for idx in range(len(subject_dirs)):
    subject_dir = subject_dirs[idx]
    print(subject_dir)
    image_path = glob.glob(os.path.join(subject_dir, '*t1n.nii.gz'))[0]
    cropped_image_path = glob.glob(os.path.join(subject_dir, '*t1n-voided.nii.gz'))[0]
    healthy_mask_path = glob.glob(os.path.join(subject_dir, '*healthy.nii.gz'))[0]
    unhealthy_mask_path = glob.glob(os.path.join(subject_dir, '*unhealthy.nii.gz'))[0]


    # 加载图像和掩膜
    image = nib.load(image_path).get_fdata().astype(
                    np.float32
                )
    healthy_mask = nib.load(healthy_mask_path).get_fdata().astype(
                    np.float32
                )
    unhealthy_mask = nib.load(unhealthy_mask_path).get_fdata().astype(
                    np.float32
                )
    cropped_image = nib.load(cropped_image_path).get_fdata().astype(
                    np.float32
                )
    image = image_preprocess(image)
    # image_copy = image
    cropped_image = image_preprocess(cropped_image)
    mask_affine = nib.load(healthy_mask_path).affine

    nonzero_coords = np.nonzero(healthy_mask)
    center_x = (np.min(nonzero_coords[0]) + np.max(nonzero_coords[0])) // 2
    center_y = (np.min(nonzero_coords[1]) + np.max(nonzero_coords[1])) // 2
    center_z = (np.min(nonzero_coords[2]) + np.max(nonzero_coords[2])) // 2
    image_shape = [240,240,155]
    img_size = 192
    # 计算裁剪区域的边界
    crop_x1 = max(center_x - int(img_size/2), 0)
    crop_x2 = min(center_x + int(img_size/2), image_shape[0])
    crop_y1 = max(center_y - int(img_size/2), 0)
    crop_y2 = min(center_y + int(img_size/2), image_shape[1])
    # crop_z1 = max(center_z - 48, 0)
    # crop_z2 = min(center_z + 48, image_shape[2])
    crop_z1 = np.min(nonzero_coords[2])
    crop_z2 = np.max(nonzero_coords[2])
    # print(crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2)

    # 如果裁剪区域小于 96x96x96,则在另一边扩展
    crop_size_x = crop_x2 - crop_x1
    crop_size_y = crop_y2 - crop_y1
    crop_size_z = crop_z2 - crop_z1
    #保存几何坐标信息
    # print(crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2)
    geometric_list = [crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2]

    if crop_size_x < img_size:
        if center_x - int(img_size/2) < 0:
            crop_x1 = 0
            crop_x2 = img_size
        else:
            crop_x1 = image_shape[0] - int(img_size)
            crop_x2 = image_shape[0]

    if crop_size_y < img_size:
        if center_y - int(img_size/2) < 0:
            crop_y1 = 0
            crop_y2 = img_size
        else:
            crop_y1 = image_shape[1] - int(img_size)
            crop_y2 = image_shape[1]

    # print(crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2)
    image = image[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]
    healthy_mask = healthy_mask[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]
    cropped_image = cropped_image[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]
    unhealthy_mask = unhealthy_mask[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]
    image_copy = image


    # 二值化掩膜
    unhealthy_mask = np.where(unhealthy_mask > 0, 1, 0)
    unhealthy_mask_bool = (unhealthy_mask == 1)
    healthy_mask = np.where(healthy_mask > 0, 1, 0)
    healthy_mask_bool = (healthy_mask == 1)

    # 使用掩码对图像进行裁剪
    image[unhealthy_mask_bool] = 0
    image_copy[healthy_mask_bool] = 0

    # 对healthy mask 部分加上图像average作为预填充
    average = np.mean(image)
    image_preinfilled = image
    image_preinfilled[healthy_mask_bool] = average

    image = image[np.newaxis, ...]
    cropped_image = cropped_image[np.newaxis, ...]
    cropped_image_preinfilled = image_preinfilled[np.newaxis, ...]
    healthy_mask = healthy_mask[np.newaxis, ...]
    image_copy = image_copy[np.newaxis, ...]

    


    # 保存每个slice的数据
    for z in range(image.shape[3]):
        slice_image = image[:, :, :, z]
        slice_adjacency_image = image[:,:,:,z-2:z+2]
        slice_healthy_mask = healthy_mask[:, :, :, z]
        slice_cropped_image = cropped_image[:, :, :, z]
        slice_cropped_image_preinfilled = cropped_image_preinfilled[:, :, :, z]
        slice_image_without_healthy_area = image_copy[:,:,:,z]
        # slice_adjacency_image = adjacency_image[:, :, :, z]

        # slice_unhealthy_mask = unhealthy_mask[:, :, z]
        
        # 保存为npy文件
        np.savez(os.path.join(subject_dir, str(img_size) + f'_slice_{z}.npz'), 
                 image=slice_image, 
                 healthy_mask=slice_healthy_mask,
                 cropped_image=slice_cropped_image,
                 cropped_image_preinfilled=slice_cropped_image_preinfilled,
                 image_without_healthy_area=slice_image_without_healthy_area,
                 geometric_list=geometric_list,
                 adjacency_image=slice_adjacency_image)