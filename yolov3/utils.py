#adarsha
#updated 24/03/2021
#updated 26/03/2021
import cv2
import time
import random
import colorsys
from yolov3.yolov3 import *

label_out = {'2': '1', '3': '2', '4': '3', '5': '4', '6': '5', '7': '6', '8': '7', '9': '8', '10': '9', '11': 'A', '12': 'B', '13': 'C', '14': 'D', '15': 'E', '16': 'F', '17': 'G', '18': 'H', '19': 'I', '20': 'J', '21': 'K', '22': 'L', '23': 'M', '24': 'N', '25': 'O', '26': 'P', '27': 'Q', '28': 'R', '29': 'S', '30': 'T', '31': 'U', '32': 'V', '33': 'W', '34': 'X', '35': 'Y', '36': 'Z'}

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence=True,
              Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # print("hsv_tuples", hsv_tuples)
    #updated 24/03/2021
    # The colorsys module defines bidirectional conversions of color values between colors expressed in the RGB (Red Green Blue)
    # color space used in computer monitors and three other coordinate systems:
    # YIQ, HLS (Hue Lightness Saturation) and HSV (Hue Saturation Value)
    #The map() function executes a specified function for each item in an iterable. The item is sent to the function as a parameter.
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

        if show_label:
            score_str = " {:.2f}".format(score) if show_confidence else ""
            if tracking: score_str = " " + str(score)
            try:
                label = "{}".format(NUM_CLASS[class_ind]) + score_str
                print('The label is {} and score (confidence) is {:.3f}'.format(label_out[NUM_CLASS[class_ind]],score))
            except KeyError:
                print("KeyError, set yolo_custom_weights = True for custom weights in configs.py")

            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX,
                                                                  fontScale, thickness=bbox_thick)
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color,
                          thickness=cv2.FILLED)
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    #:param bboxes: (xmin, ymin, xmax, ymax, score, class)
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def detect_image(Yolo, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                 score_threshold=0.3, iou_threshold=0.98, rectangle_colors=''):
    original_image = cv2.imread(image_path)
    #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = Yolo.predict(image_data)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)

    if output_path != '':
        cv2.imwrite(output_path, image)
    if show:
        cv2.imshow("predicted image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image

# detect from webcam
def detect_webcam(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.98,
                    iou_threshold=0.3, rectangle_colors=''):
    times = []
    vid = cv2.VideoCapture(video_path)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)  # output_path must be .mp4

    while True:
        ret, original_frame = vid.read()
        if not ret:
            print('Cannot get video/webcam frame. Exiting')
            break

        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        tic = time.time()
        pred_bbox = Yolo.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')
        toe = time.time()
        times.append(toe - tic)
        times = times[-20:]
        ms = sum(times) / len(times) * 1000
        fps = 1000 / ms
        print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

        frame = draw_bbox(original_frame, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        cv2.putText(frame, "Time: {:.1f}FPS".format(fps), (0, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

        if output_path != '':
            out.write(frame)
        if show:
            cv2.imshow('output', frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
