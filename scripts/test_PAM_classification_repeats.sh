#!/bin/sh
#SBATCH -o PAM-classification-%j.output
#SBATCH -p K20q
#SBATCH -n 4
export PATH=/home/yuhao001/anaconda3/envs/mos2_project/bin:$PATH
cd /home/yuhao001/projects/ML-guided-material-synthesis/
python3 PAM-classification_repeat1000times.py
