import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from yolov3.utils import detect_image, detect_webcam,Load_Yolo_model, detect_video
from yolov3.configs import *

image_path   = "./IMAGES/city.jpg"
video_path = 'C:/Users/aryal/Desktop/real-data-test/input/real-test-4.mp4'
video_output ='C:/Users/aryal/Desktop/real-data-test/output/real-test-4-output.mp4'

yolo = Load_Yolo_model()
#detect_image(yolo, image_path, "./IMAGES/city_out.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(0,0,0))
detect_video(yolo, video_path, video_output, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_webcam(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(0,0,255))



