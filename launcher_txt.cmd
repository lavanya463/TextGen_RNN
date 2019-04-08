#!/bin/bash

#SBATCH --job-name="test_text_gen"

#SBATCH --workdir=.

#SBATCH --output=textgen_%j.out

#SBATCH --error=textgen_%j.err

#SBATCH --ntasks=1

#SBATCH --gres gpu:1

#SBATCH --time=7:00:00

#SBATCH --cpus-per-task=4

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python text_generation_RNN.py


