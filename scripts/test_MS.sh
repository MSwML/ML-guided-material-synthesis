#!/bin/sh
#SBATCH -o IPYTNB-%j.output
#SBATCH -p K20q
#SBATCH -n 4
export PATH=/home/yuhao001/anaconda3/envs/mos2_project/bin:$PATH
cd /home/yuhao001/projects/ML-guided-material-synthesis/


# execute all notebooks in parallel
for f in *.ipynb
do
    echo "$f is running"
    jupyter nbconvert \
      --ExecutePreprocessor.allow_errors=True \
      --ExecutePreprocessor.timeout=-1 \
      --FilesWriter.build_directory=./results \
      --execute "$f" &
done

wait