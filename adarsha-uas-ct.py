
from imutils import paths
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
folder_name = 'yolo-custom-test'
output_folder = 'yolo-custom-output'

from yolov3.utils import detect_image, detect_webcam, Load_Yolo_model
from yolov3.configs import *
yolo = Load_Yolo_model()
i = 0
for imagePath in sorted(paths.list_images(folder_name)):
    detect_image(yolo, imagePath, "./{}/{}_out.jpg".format(output_folder,i), input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(0,0,0))
    i+=1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyAllWindows()

#detect_webcam(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(0,255, 0))

