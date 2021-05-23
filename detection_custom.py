import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from yolov3.utils import detect_image, detect_webcam,Load_Yolo_model
from yolov3.configs import *

webcam = True
image_path = "./IMAGES/city.jpg"

video_path = 0 if webcam else './real-data-test/input/real-test-4.mp4'

video_output = "" if webcam else './real-data-test/output/real_test_NX.mp4'
show = True if webcam else False

yolo = Load_Yolo_model()
#detect_image(yolo, image_path, "./IMAGES/city_out.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(0,0,0))
detect_webcam(yolo, video_path, video_output, input_size=YOLO_INPUT_SIZE, show=show, CLASSES=TRAIN_CLASSES, rectangle_colors=(0,0,255))

