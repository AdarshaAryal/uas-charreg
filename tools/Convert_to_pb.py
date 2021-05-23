#adarsha
#updated 28/02/2021
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys


foldername = os.path.basename(os.getcwd())
if foldername == "tools":
    os.chdir("..")
sys.path.insert(1, os.getcwd())

from yolov3.utils import load_yolo_weights
from yolov3.configs import *
from yolov3.yolov3 import *

Darknet_weights =  yolov3_weights

if yolo_custom_weights == False:
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
    load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
else:
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.load_weights(yolo_custom_weights) # use custom weights

yolo.summary()
yolo.save(f'./checkpoints/{yolo_type}-{YOLO_INPUT_SIZE}')

print(f"model saves to /checkpoints/{yolo_type}-{YOLO_INPUT_SIZE}")
