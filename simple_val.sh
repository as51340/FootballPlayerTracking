# ------------------------------------------------------------- #
# Script for starting validation of YOLOv5 model.               # 
# ------------------------------------------------------------- #

python val.py --img 3264 --batch 4 --data ../data.yaml --weights runs/train/exp12/weights/best.pt --task test --save-txt
