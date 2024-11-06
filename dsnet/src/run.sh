#!/bin/bash
module load compilers/cuda/11.6
module load cudnn/8.4.0.27_cuda11.x
module load compilers/gcc/9.3.0
source activate dsnet
export PYTHONUNBUFFERED=1
python -u train.py