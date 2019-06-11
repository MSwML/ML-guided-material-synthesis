#!/bin/sh


# setup directory 
#cd /XXX/ML-guided-material-synthesis/


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