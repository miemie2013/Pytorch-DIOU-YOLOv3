#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Author      : miemie2013
#   Created date: 2020-01-11 16:31:57
#   Description : pytorch_yolov3
#
#================================================================
import cv2
import sys
import time
import torch
import random
import numpy as np
import os
import platform
from model.darknet_yolo_pt import Darknet, YoloLoss

sysstr = platform.system()
use_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())
print(torch.__version__)
# 禁用cudnn就能解决Windows报错问题。Windows用户如果删掉之后不报CUDNN_STATUS_EXECUTION_FAILED，那就可以删掉。
if sysstr == 'Windows':
    torch.backends.cudnn.enabled = False

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

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

def image_preporcess(image, target_size, gt_boxes=None):
    # 这里改变了一部分原作者的代码。可以发现，传入训练的图片是bgr格式
    ih, iw = target_size
    h, w = image.shape[:2]
    M, h_out, w_out = training_transform(h, w, ih, iw)
    # 填充黑边缩放
    letterbox = cv2.warpAffine(image, M, (w_out, h_out))
    pimage = np.float32(letterbox) / 255.
    if gt_boxes is None:
        return pimage
    else:
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return pimage, gt_boxes

def random_fill(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        # 水平方向填充黑边，以训练小目标检测
        if random.random() < 0.5:
            dx = random.randint(int(0.5*w), int(1.5*w))
            black_1 = np.zeros((h, dx, 3), dtype='uint8')
            black_2 = np.zeros((h, dx, 3), dtype='uint8')
            image = np.concatenate([black_1, image, black_2], axis=1)
            bboxes[:, [0, 2]] += dx
        # 垂直方向填充黑边，以训练小目标检测
        else:
            dy = random.randint(int(0.5*h), int(1.5*h))
            black_1 = np.zeros((dy, w, 3), dtype='uint8')
            black_2 = np.zeros((dy, w, 3), dtype='uint8')
            image = np.concatenate([black_1, image, black_2], axis=0)
            bboxes[:, [1, 3]] += dy
    return image, bboxes

def random_horizontal_flip(image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0,2]] = w - bboxes[:, [2,0]]
    return image, bboxes

def random_crop(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return image, bboxes

def random_translate(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return image, bboxes

def parse_annotation(annotation, train_input_size, annotation_type):
    line = annotation.split()
    image_path = line[0]
    # image_path = '../'+line[0]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = np.array(cv2.imread(image_path))
    # 没有标注物品，即每个格子都当作背景处理
    exist_boxes = True
    if len(line) == 1:
        bboxes = np.array([[10, 10, 101, 103, 0]])
        exist_boxes = False
    else:
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
    if annotation_type == 'train':
        # image, bboxes = random_fill(np.copy(image), np.copy(bboxes))    # 数据集缺乏小物体时打开
        image, bboxes = random_horizontal_flip(np.copy(image), np.copy(bboxes))
        image, bboxes = random_crop(np.copy(image), np.copy(bboxes))
        image, bboxes = random_translate(np.copy(image), np.copy(bboxes))
    image, bboxes = image_preporcess(np.copy(image), [train_input_size, train_input_size], np.copy(bboxes))
    return image, bboxes, exist_boxes

def bbox_iou_data(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return inter_area / union_area

def preprocess_true_boxes(bboxes, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors):
    label = [np.zeros((train_output_sizes[i], train_output_sizes[i], 3,
                       5 + num_classes)) for i in range(3)]
    bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(3)]
    bbox_count = np.zeros((3,))
    for bbox in bboxes:
        bbox_coor = bbox[:4]
        bbox_class_ind = bbox[4]
        onehot = np.zeros(num_classes, dtype=np.float)
        onehot[bbox_class_ind] = 1.0
        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
        iou = []
        for i in range(3):
            anchors_xywh = np.zeros((3, 4))
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = anchors[i]
            iou_scale = bbox_iou_data(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
        best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
        best_detect = int(best_anchor_ind / 3)
        best_anchor = int(best_anchor_ind % 3)
        xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
        # 防止越界
        grid_r = label[best_detect].shape[0]
        grid_c = label[best_detect].shape[1]
        xind = max(0, xind)
        yind = max(0, yind)
        xind = min(xind, grid_r-1)
        yind = min(yind, grid_c-1)
        label[best_detect][yind, xind, best_anchor, :] = 0
        label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
        label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
        label[best_detect][yind, xind, best_anchor, 5:] = onehot
        bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
        bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
        bbox_count[best_detect] += 1
    label_sbbox, label_mbbox, label_lbbox = label
    sbboxes, mbboxes, lbboxes = bboxes_xywh
    return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

def generate_one_batch(annotation_lines, step, batch_size, anchors, num_classes, max_bbox_per_scale, annotation_type):
    n = len(annotation_lines)

    # 多尺度训练
    train_input_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
    train_input_size = random.choice(train_input_sizes)
    strides = np.array([8, 16, 32])

    # 输出的网格数
    train_output_sizes = train_input_size // strides

    batch_image = np.zeros((batch_size, train_input_size, train_input_size, 3))

    batch_label_sbbox = np.zeros((batch_size, train_output_sizes[0], train_output_sizes[0],
                                  3, 5 + num_classes))
    batch_label_mbbox = np.zeros((batch_size, train_output_sizes[1], train_output_sizes[1],
                                  3, 5 + num_classes))
    batch_label_lbbox = np.zeros((batch_size, train_output_sizes[2], train_output_sizes[2],
                                  3, 5 + num_classes))

    batch_sbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
    batch_mbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
    batch_lbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))

    if (step+1)*batch_size > n:
        batch = annotation_lines[n-batch_size:n]
    else:
        batch = annotation_lines[step*batch_size:(step+1)*batch_size]
    for num in range(batch_size):
        image, bboxes, exist_boxes = parse_annotation(batch[num], train_input_size, annotation_type)
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = preprocess_true_boxes(bboxes, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors)

        batch_image[num, :, :, :] = image
        if exist_boxes:
            batch_label_sbbox[num, :, :, :, :] = label_sbbox
            batch_label_mbbox[num, :, :, :, :] = label_mbbox
            batch_label_lbbox[num, :, :, :, :] = label_lbbox
            batch_sbboxes[num, :, :] = sbboxes
            batch_mbboxes[num, :, :] = mbboxes
            batch_lbboxes[num, :, :] = lbboxes
    batch_image = batch_image.transpose(0, 3, 1, 2)
    return batch_image, [batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes]

if __name__ == '__main__':
    # train_path = 'annotation/voc2007_train.txt'
    # val_path = 'annotation/voc2007_val.txt'
    # classes_path = 'data/voc_classes.txt'

    train_path = 'annotation/coco2017_train.txt'
    val_path = 'annotation/coco2017_val.txt'
    classes_path = 'data/coco_classes.txt'

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = np.array([
        [[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],
        [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],
        [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]
    ])

    # 模式。 0-从头训练，1-读取模型训练（包括解冻），2-读取coco预训练模型训练
    pattern = 0
    save_best_only = False
    max_bbox_per_scale = 150
    iou_loss_thresh = 0.7
    
    # 经过试验发现，使用focal_loss会增加误判fp，所以默认使用二值交叉熵损失函数训练。下面这3个alpha请忽略。
    # 经过试验发现alpha取>0.5的值时mAP会提高，但误判（False Predictions）会增加；alpha取<0.5的值时mAP会降低，误判会降低。
    # 试验时alpha_1取0.95，alpha_2取0.85，alpha_3取0.75
    # 小感受野输出层输出的格子最多，预测框最多，正样本很有可能占比是最少的，所以试验时alpha_1 > alpha_2 > alpha_3
    alpha_1 = 0.5    # 小感受野输出层的focal_loss的alpha
    alpha_2 = 0.5    # 中感受野输出层的focal_loss的alpha
    alpha_3 = 0.5    # 大感受野输出层的focal_loss的alpha

    # 初始卷积核个数
    initial_filters = 8

    net = Darknet(num_classes, initial_filters=initial_filters)
    if pattern == 2:
        lr = 0.0001
        batch_size = 8
        initial_epoch = 0
        epochs = 999
        # 冻结代码待补充
        # 分支2还未完成

        net.load_state_dict(torch.load('yolo_bgr_mAP_47.pt'))
    elif pattern == 1:
        lr = 0.000001
        batch_size = 6
        initial_epoch = 20
        epochs = 50
        # 解冻代码待补充
        # 分支1可用

        net.load_state_dict(torch.load('ep000006-loss1.095-val_loss0.872.pt'))
    elif pattern == 0:
        lr = 0.0001
        batch_size = 6
        initial_epoch = 0
        epochs = 130

    # 打印网络结构
    # print(net)
    device = torch.device('cuda' if use_cuda else 'cpu')
    net_img = net.to(device)
    from torchsummary import summary
    summary(net_img, (3, 416, 416))

    # 建立损失函数
    yolo_loss = YoloLoss(num_classes, iou_loss_thresh, anchors, alpha_1, alpha_2, alpha_3)
    if use_cuda:
        yolo_loss = yolo_loss.cuda()  # 如果有gpu可用，损失函数存放在gpu显存里
        net = net.cuda()              # 如果有gpu可用，模型（包括了权重weight）存放在gpu显存里

    # 验证集和训练集
    with open(train_path) as f:
        train_lines = f.readlines()
    with open(val_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 一轮的步数
    train_steps = int(num_train / batch_size) if num_train % batch_size == 0 else int(num_train / batch_size)+1
    val_steps = int(num_val / batch_size) if num_val % batch_size == 0 else int(num_val / batch_size)+1
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 传入 net 的所有参数, 学习率

    best_val_loss = 0.0
    for t in range(initial_epoch, epochs, 1):
        print('Epoch %d/%d\n'%(t+1, epochs))
        epochStartTime = time.time()
        start = time.time()
        # 每个epoch之前洗乱
        np.random.shuffle(train_lines)
        train_epoch_loss, val_epoch_loss = [], []

        # 训练阶段
        net.train()
        for step in range(train_steps):
            batch_image, lables = generate_one_batch(train_lines, step, batch_size, anchors, num_classes, max_bbox_per_scale, 'train')
            if use_cuda:
                batch_image = torch.Tensor(batch_image).cuda()
                lables = [torch.Tensor(it).cuda() for it in lables]
            else:
                batch_image = torch.Tensor(batch_image)
                lables = [torch.Tensor(it) for it in lables]
            y1_pred, y2_pred, y3_pred = net(batch_image)   # 直接卷积后的输出
            args = [y1_pred, y2_pred, y3_pred] + lables
            train_step_loss = yolo_loss(args)
            step_loss = 0.
            if use_cuda:
                step_loss = train_step_loss.cpu().data.numpy()
            else:
                step_loss = train_step_loss.data.numpy()
            train_epoch_loss.append(step_loss)

            # 自定义进度条，高仿keras
            percent = ((step + 1) / train_steps) * 100
            num = int(29 * percent / 100)
            time.sleep(0.1)
            ETA = int((time.time() - epochStartTime) * (100 - percent) / percent)
            sys.stdout.write('\r{0}'.format(' ' * (len(str(train_steps)) - len(str(step + 1)))) + \
                             '{0}/{1} [{2}>'.format(step + 1, train_steps, '=' * num) + '{0}'.format('.' * (29 - num)) + ']' + \
                             ' - ETA: ' + str(ETA) + 's' + ' - loss: %.4f'%(step_loss, ))
            sys.stdout.flush()

            # 更新权重
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            train_step_loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

        # 验证阶段
        net.eval()
        for step in range(val_steps):
            batch_image, lables = generate_one_batch(val_lines, step, batch_size, anchors, num_classes, max_bbox_per_scale, 'val')
            if use_cuda:
                batch_image = torch.Tensor(batch_image).cuda()
                lables = [torch.Tensor(it).cuda() for it in lables]
            else:
                batch_image = torch.Tensor(batch_image)
                lables = [torch.Tensor(it) for it in lables]
            y1_pred, y2_pred, y3_pred = net(batch_image)   # 直接卷积后的输出
            args = [y1_pred, y2_pred, y3_pred] + lables
            val_step_loss = yolo_loss(args)
            step_loss = 0.
            if use_cuda:
                step_loss = val_step_loss.cpu().data.numpy()
            else:
                step_loss = val_step_loss.data.numpy()
            val_epoch_loss.append(step_loss)

        train_epoch_loss, val_epoch_loss = np.mean(train_epoch_loss), np.mean(val_epoch_loss)

        # 保存模型
        content = '%d\tloss = %.4f\tval_loss = %.4f\n' % ((t + 1), train_epoch_loss, val_epoch_loss)
        with open('yolov3_pytorch_logs.txt', 'a', encoding='utf-8') as f:
            f.write(content)
            f.close()
        path_dir = os.listdir('./')
        eps = []
        names = []
        for name in path_dir:
            if name[len(name) - 2:len(name)] == 'pt' and name[0:2] == 'ep':
                sss = name.split('-')
                ep = int(sss[0][2:])
                eps.append(ep)
                names.append(name)
        if len(eps) >= 10:
            i2 = eps.index(min(eps))
            os.remove(names[i2])
        if t == initial_epoch:
            best_val_loss = val_epoch_loss
            torch.save(net.state_dict(),
                       'ep%.6d-loss%.3f-val_loss%.3f.pt' % ((t + 1), train_epoch_loss, val_epoch_loss))
        else:
            if save_best_only:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    torch.save(net.state_dict(),
                               'ep%.6d-loss%.3f-val_loss%.3f.pt' % ((t + 1), train_epoch_loss, val_epoch_loss))
            else:
                torch.save(net.state_dict(),
                           'ep%.6d-loss%.3f-val_loss%.3f.pt' % ((t + 1), train_epoch_loss, val_epoch_loss))

        # 打印本轮训练结果
        sys.stdout.write(
            '\r{0}/{1} [{2}='.format(train_steps, train_steps, '=' * num) + '{0}'.format('.' * (29 - num)) + ']' + \
            ' - %ds' % (int(time.time() - epochStartTime),) + ' - loss: %.4f'%(train_epoch_loss, ) + ' - val_loss: %.4f\n'%(val_epoch_loss, ))
        sys.stdout.flush()

