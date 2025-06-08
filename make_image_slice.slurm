#!/bin/bash
#SBATCH -J bcp_age_predict
#SBATCH -p bme_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=fail
#SBATCH --mail-user=taotl2022@shanghaitech.edu.cn
#SBATCH -N 1
#SBATCH -t 5-00:00:00
#SBATCH --output=slice.out
#SBATCH --error=slice.error

pwd
hostname
date
module list

conda env list
source /hpc/data/home/bme/yubw/anaconda3/bin/activate ddpm

python3 make_image_slice.py

date