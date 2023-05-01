# ------------------------------------------------------------- #
# Script for starting training of YOLOv5 model on the specified #
# baseline dataset.                                             #
# ------------------------------------------------------------- #
python train.py --img 3264 --batch 4 --epochs 300 --data ../data.yaml --weights "yolov5n.pt" --cache
