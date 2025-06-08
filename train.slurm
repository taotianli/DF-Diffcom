#!/bin/bash
#SBATCH -J bcp_age_predict
#SBATCH -p bme_gpu
#SBATCH --exclude=bme_gpu01,bme_gpu02,bme_gpu03,bme_gpu04,bme_gpu05,bme_gpu06,bme_gpu07,bme_gpu08,bme_gpu10
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=fail
#SBATCH --mail-user=taotl2022@shanghaitech.edu.cn
#SBATCH -N 1
#SBATCH -t 5-00:00:00
#SBATCH --gres=gpu
#SBATCH --output=train.out
#SBATCH --error=train.error

pwd
hostname
date
module list
module add cuda/7/11.4

nvcc -V
conda env list
source /hpc/data/home/bme/yubw/anaconda3/bin/activate ddpm
nvidia-smi
lspci|grep -i vga

python3 ddpm_conditional.py

date