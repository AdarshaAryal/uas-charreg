# adarsha
# updated 15/02/2021
# updated 15/03/2021

# YOLO options
yolo_type = "yolov3"
yolov3_weights = "./model_data/yolo_weights/yolov3.weights"
#yolo_custom_weights = "checkpoints/yolov3_custom.data-00000-of-00001"  # if not using leave False
yolo_custom_weights = "checkpoints/yolov3_custom"
#yolo_custom_weights = False
YOLO_COCO_CLASSES = "./model_data/coco/coco.names"
YOLO_STRIDES = [8, 16, 32]
YOLO_IOU_LOSS_THRESH = 0.5
YOLO_ANCHOR_PER_SCALE = 3
YOLO_MAX_BBOX_PER_SCALE = 100
YOLO_INPUT_SIZE = 416

# yolo v3 anchors
# anchor boxes can be used to detect multiple objects in the same grid.
# However, for our purpose, we will not require this. As there's only one alphanumeric character in a grid.
# They are hand-picked using experimentation but a better version is to use k-means algorithm

YOLO_ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]]
# Train options
# TRAIN_YOLO_TINY = False
TRAIN_SAVE_BEST_ONLY = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT = False  # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_CLASSES = "./model_data/chars_subset_labels.txt"
# TRAIN_CLASSES = "./model_data/coco/coco.names"

TRAIN_ANNOTATION_PATH = "./model_data/chars_subset_train.txt"
# TRAIN_ANNOTATION_PATH = "./model_data/coco/train2017.txt"
TRAIN_LOGDIR = "log"
TRAIN_CHECKPOINTS_FOLDER = "checkpoints"
TRAIN_MODEL_NAME = f"{yolo_type}_custom"
TRAIN_LOAD_IMAGES_TO_RAM = True  # With True faster training, but need more RAM
TRAIN_BATCH_SIZE = 1
TRAIN_INPUT_SIZE = 416
TRAIN_DATA_AUG = True
TRANSFER_LEARNING = True
TRAIN_FROM_CHECKPOINT = False  # "checkpoints/yolov3_custom"
TRAIN_LR_INIT = 1e-4
TRAIN_LR_END = 1e-6
TRAIN_WARMUP_EPOCHS = 0
TRAIN_EPOCHS = 50

# TEST options
TEST_ANNOTATION_PATH = "./model_data/chars_subset_test.txt"
# TEST_ANNOTATION_PATH = "./model_data/coco/val2017.txt"
TEST_BATCH_SIZE = 1
TEST_INPUT_SIZE = 416
# never use data augmentation on  test data
TEST_DATA_AUG = False
TEST_DETECTED_IMAGE_PATH = ""
TEST_SCORE_THRESHOLD = 0.5  # was 0.3 before
TEST_IOU_THRESHOLD = 0.45
