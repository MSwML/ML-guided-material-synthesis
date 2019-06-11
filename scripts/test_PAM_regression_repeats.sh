#!/bin/sh
#SBATCH -o PAM-regression-%j.output
#SBATCH -p RTXq
#SBATCH -n 4
export PATH=/home/yuhao001/anaconda3/envs/mos2_project/bin:$PATH
cd /home/yuhao001/projects/ML-guided-material-synthesis/
python3 PAM-regression_repeat1000times.py
