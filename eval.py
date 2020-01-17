#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================
import colorsys
import random

import cv2
import time
import os
import shutil
import numpy as np

from model.decode_pt import Decode

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)



def draw_bbox(image, bboxes, classes, show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = 1 if min(image_h, image_w) < 400 else 2
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)
    return image

def training_transform(height, width, output_height, output_width):
    height_scale, width_scale = output_height / height, output_width / width
    scale = min(height_scale, width_scale)
    resize_height, resize_width = round(height * scale), round(width * scale)
    pad_top = (output_height - resize_height) // 2
    pad_left = (output_width - resize_width) // 2
    A = np.float32([[scale, 0.0], [0.0, scale]])
    B = np.float32([[pad_left], [pad_top]])
    M = np.hstack([A, B])
    return M, output_height, output_width

def process_image(img, input_shape):
    h, w = img.shape[:2]
    M, h_out, w_out = training_transform(h, w, input_shape[0], input_shape[1])
    # 填充黑边缩放
    letterbox = cv2.warpAffine(img, M, (w_out, h_out))
    pimage = np.float32(letterbox) / 255.
    pimage = np.expand_dims(pimage, axis=0)
    return pimage

def detect_image(image, _decode, input_shape):
    pimage = process_image(image, input_shape)

    start = time.time()
    boxes, scores, classes = _decode.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.6f}s'.format(end - start))
    if boxes is not None:
        bboxes = []
        for i in range(len(boxes)):
            bbox = []
            bbox.append(boxes[i][0])
            bbox.append(boxes[i][1])
            bbox.append(boxes[i][2])
            bbox.append(boxes[i][3])
            bbox.append(scores[i])
            bbox.append(classes[i])
            bbox = np.array(bbox)
            bboxes.append(bbox)
        return bboxes
    return []


class YoloTest(object):
    def __init__(self):
        self.input_shape       = (416, 416)

        # COCO
        self.file             = 'data/coco_classes.txt'
        self.annotation_path  = 'annotation/coco2017_val.txt'

        # VOC
        # self.file             = 'data/voc_classes.txt'
        # self.annotation_path  = 'annotation/voc2007_val.txt'

        self.classes          = read_class_names(self.file)

        # 是否保存画框的照片
        # self.write_image      = False
        self.write_image      = True

        self.write_image_path = "./mAP/detection/"
        self.show_label       = True
        self.num_classes      = len(self.classes)

        # 只用pytorch
        self._decode = Decode(0.3, 0.45, self.input_shape, 'yolo_bgr_mAP_47.pt', self.file, initial_filters=32)



    def evaluate(self):
        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        with open(self.annotation_path, 'r') as annotation_file:
            start = time.time()
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt=[]
                    classes_gt=[]
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        # print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                bboxes_pr = detect_image(image, self._decode, self.input_shape)

                if self.write_image:
                    image = draw_bbox(image, bboxes_pr, self.classes, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path+image_name, image)

                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
            print('total time: {0:.6f}s'.format(time.time() - start))

import torch
import platform
sysstr = platform.system()
print(torch.cuda.is_available())
print(torch.__version__)
# 禁用cudnn就能解决Windows报错问题。Windows用户如果删掉之后不报CUDNN_STATUS_EXECUTION_FAILED，那就可以删掉。
if sysstr == 'Windows':
    torch.backends.cudnn.enabled = False


if __name__ == '__main__': YoloTest().evaluate()



