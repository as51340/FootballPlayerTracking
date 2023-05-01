# ----------------------------------------------------------------------------- #
# This script is used to evaluate the pretrained models on the validation set.  #
# ----------------------------------------------------------------------------- #

#!/bin/bash

models=("yolov5s.pt" "yolov5n.pt" "yolov5m.pt" "yolov5l.pt" "yolov5x.pt" 
  "yolov5s6.pt" "yolov5n6.pt" "yolov5m6.pt" "yolov5l6.pt" "yolov5x6.pt"
)
for model in ${models[@]};
do
  python3 val.py --weights $model --data ../data.yaml --single-cls --img 3260
done
