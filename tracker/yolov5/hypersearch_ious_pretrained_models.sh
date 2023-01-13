#!/bin/bash

ious=(0.34 0.6 0.8 0.95)
for iou in ${ious[@]};
do
  python3 val.py --weights yolov5s.pt --data ../data.yaml --single-cls --img 3260 --iou-thres $iou
done
