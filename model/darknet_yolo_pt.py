#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Author      : miemie2013
#   Created date: 2020-01-11 16:31:57
#   Description : pytorch_yolov3
#
#================================================================
import torch
import torch as T
import math
import numpy as np

def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:

    举例时假设pred_xywh和label_xywh的shape都是(1, 4)
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = T.cat((boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5), dim=-1)
    boxes2_x0y0x1y1 = T.cat((boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5), dim=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = T.cat((T.min(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                             T.max(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])), dim=-1)
    boxes2_x0y0x1y1 = T.cat((T.min(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                             T.max(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])), dim=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = T.max(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = T.min(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = right_down - left_up
    inter_section = T.where(inter_section < 0.0, inter_section*0, inter_section)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = T.min(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = T.max(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = T.pow(enclose_wh[..., 0], 2) + T.pow(enclose_wh[..., 1], 2)

    # 两矩形中心点距离的平方
    p2 = T.pow(boxes1[..., 0] - boxes2[..., 0], 2) + T.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # 增加av。加上除0保护防止nan。
    atan1 = T.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-9))
    atan2 = T.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-9))
    v = 4.0 * T.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou


def bbox_iou(boxes1, boxes2):
    '''
    预测框          boxes1 (?, grid_h, grid_w, 3,   1, 4)，神经网络的输出(tx, ty, tw, th)经过了后处理求得的(bx, by, bw, bh)
    图片中所有的gt  boxes2 (?,      1,      1, 1, 150, 4)
    '''
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # 所有格子的3个预测框的面积
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]  # 所有ground truth的面积

    # (x, y, w, h)变成(x0, y0, x1, y1)
    boxes1 = T.cat((boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                    boxes1[..., :2] + boxes1[..., 2:] * 0.5), dim=-1)
    boxes2 = T.cat((boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                    boxes2[..., :2] + boxes2[..., 2:] * 0.5), dim=-1)

    # 所有格子的3个预测框 分别 和  150个ground truth  计算iou。 所以left_up和right_down的shape = (?, grid_h, grid_w, 3, 150, 2)
    left_up = T.max(boxes1[..., :2], boxes2[..., :2])  # 相交矩形的左上角坐标
    right_down = T.min(boxes1[..., 2:], boxes2[..., 2:])  # 相交矩形的右下角坐标

    # 相交矩形的w和h，是负数时取0     (?, grid_h, grid_w, 3, 150, 2)
    inter_section = right_down - left_up
    inter_section = T.where(inter_section < 0.0, inter_section*0, inter_section)
    inter_area = inter_section[..., 0] * inter_section[..., 1]  # 相交矩形的面积            (?, grid_h, grid_w, 3, 150)
    union_area = boxes1_area + boxes2_area - inter_area  # union_area      (?, grid_h, grid_w, 3, 150)
    iou = 1.0 * inter_area / union_area  # iou                             (?, grid_h, grid_w, 3, 150)
    return iou

def loss_layer(conv, pred, label, bboxes, stride, num_class, iou_loss_thresh, alpha=0.5, gamma=2):
    conv_shape = conv.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]
    pred_prob = pred[:, :, :, :, 5:]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    ciou = bbox_ciou(pred_xywh, label_xywh)                             # (8, 13, 13, 3)
    ciou = ciou.reshape((batch_size, output_size, output_size, 3, 1))   # (8, 13, 13, 3, 1)
    input_size = float(input_size)

    # 每个预测框xxxiou_loss的权重 = 2 - (ground truth的面积/图片面积)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)  # 1. respond_bbox作为mask，有物体才计算xxxiou_loss

    # 2. respond_bbox作为mask，有物体才计算类别loss
    prob_pos_loss = label_prob * (0 - T.log(pred_prob + 1e-9))             # 二值交叉熵，tf中也是加了极小的常数防止nan
    prob_neg_loss = (1 - label_prob) * (0 - T.log(1 - pred_prob + 1e-9))   # 二值交叉熵，tf中也是加了极小的常数防止nan
    prob_mask = respond_bbox.repeat((1, 1, 1, 1, num_class))
    prob_loss = prob_mask * (prob_pos_loss + prob_neg_loss)

    # 3. xxxiou_loss和类别loss比较简单。重要的是conf_loss，是一个focal_loss
    # 分两步：第一步是确定 grid_h * grid_w * 3 个预测框 哪些作为反例；第二步是计算focal_loss。
    expand_pred_xywh = pred_xywh[:, :, :, :, np.newaxis, :]  # 扩展为(?, grid_h, grid_w, 3,   1, 4)
    expand_bboxes = bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]  # 扩展为(?,      1,      1, 1, 150, 4)
    iou = bbox_iou(expand_pred_xywh, expand_bboxes)  # 所有格子的3个预测框 分别 和  150个ground truth  计算iou。   (?, grid_h, grid_w, 3, 150)
    max_iou, max_iou_indices = T.max(iou, dim=-1, keepdim=True)        # 与150个ground truth的iou中，保留最大那个iou。  (?, grid_h, grid_w, 3, 1)

    # respond_bgd代表  这个分支输出的 grid_h * grid_w * 3 个预测框是否是 反例（背景）
    # label有物体，respond_bgd是0。 没物体的话：如果和某个gt(共150个)的iou超过iou_loss_thresh，respond_bgd是0；如果和所有gt(最多150个)的iou都小于iou_loss_thresh，respond_bgd是1。
    # respond_bgd是0代表有物体，不是反例；  权重respond_bgd是1代表没有物体，是反例。
    # 有趣的是，模型训练时由于不断更新，对于同一张图片，两次预测的 grid_h * grid_w * 3 个预测框（对于这个分支输出）  是不同的。用的是这些预测框来与gt计算iou来确定哪些预测框是反例。
    # 而不是用固定大小（不固定位置）的先验框。
    respond_bgd = (1.0 - respond_bbox) * (max_iou < iou_loss_thresh).float()

    # focal_loss介绍： https://www.cnblogs.com/king-lps/p/9497836.html  公式简单，但是效果出群！alpha解决不平衡问题，gamma解决困难样本问题。
    # 为什么正样本数量少，给的权重alpha比负样本的权重(1-alpha)还小？ 请看 https://blog.csdn.net/weixin_44638957/article/details/100733971

    # YunYang1994的focal_loss，只带gamma解决困难样本问题。没有带上alpha。
    # pos_loss = respond_bbox * (0 - T.log(pred_conf + 1e-9)) * T.pow(1 - pred_conf, gamma)
    # neg_loss = respond_bgd  * (0 - T.log(1 - pred_conf + 1e-9)) * T.pow(pred_conf, gamma)

    # RetinaNet的focal_loss，多带上alpha解决不平衡问题。
    # 经过试验发现alpha取>0.5的值时mAP会提高，但误判（False Predictions）会增加；alpha取<0.5的值时mAP会降低，误判会降低。
    # pos_loss = respond_bbox * (0 - T.log(pred_conf + 1e-9)) * T.pow(1 - pred_conf, gamma) * alpha
    # neg_loss = respond_bgd  * (0 - T.log(1 - pred_conf + 1e-9)) * T.pow(pred_conf, gamma) * (1 - alpha)

    # 二值交叉熵损失
    pos_loss = respond_bbox * (0 - T.log(pred_conf + 1e-9))
    neg_loss = respond_bgd  * (0 - T.log(1 - pred_conf + 1e-9))

    conf_loss = pos_loss + neg_loss
    # 回顾respond_bgd，某个预测框和某个gt的iou超过iou_loss_thresh，不被当作是反例。在参与“预测的置信位 和 真实置信位 的 二值交叉熵”时，这个框也可能不是正例(label里没标这个框是1的话)。这个框有可能不参与置信度loss的计算。
    # 这种框一般是gt框附近的框，或者是gt框所在格子的另外两个框。它既不是正例也不是反例不参与置信度loss的计算，其实对yolov3算法是有好处的。（论文里称之为ignore）
    # 它如果作为反例参与置信度loss的计算，会降低yolov3的精度。
    # 它如果作为正例参与置信度loss的计算，可能会导致预测的框不准确（因为可能物体的中心都预测不准）。

    ciou_loss = ciou_loss.sum((1, 2, 3, 4)).mean()    # 每个样本单独计算自己的ciou_loss，再求平均值
    conf_loss = conf_loss.sum((1, 2, 3, 4)).mean()    # 每个样本单独计算自己的conf_loss，再求平均值
    prob_loss = prob_loss.sum((1, 2, 3, 4)).mean()    # 每个样本单独计算自己的prob_loss，再求平均值

    return ciou_loss + conf_loss + prob_loss

def get_grid_offset(grid_n):
    grid_offset = np.arange(grid_n)
    grid_x_offset = np.tile(grid_offset, (grid_n, 1))
    grid_y_offset = np.copy(grid_x_offset)
    grid_y_offset = grid_y_offset.transpose(1, 0)
    grid_x_offset = np.reshape(grid_x_offset, (grid_n, grid_n, 1, 1))
    grid_x_offset = np.tile(grid_x_offset, (1, 1, 3, 1))
    grid_y_offset = np.reshape(grid_y_offset, (grid_n, grid_n, 1, 1))
    grid_y_offset = np.tile(grid_y_offset, (1, 1, 3, 1))
    grid_offset = np.concatenate([grid_x_offset, grid_y_offset], axis=-1)
    return grid_offset

def decode(conv_output, anchors, stride):
    conv_shape = conv_output.shape
    output_size = conv_shape[1]

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    grid_offset = get_grid_offset(output_size)

    # pytorch支持张量Tensor和标量相加（相乘），而不支持张量Tensor和同shape或不同shape的ndarray相加（相乘）。
    # pytorch支持张量Tensor和同shape或不同shape的Tensor相加（相乘）。
    grid_offset = torch.Tensor(grid_offset.astype(np.float32))
    anchor_t = torch.Tensor(np.copy(anchors).astype(np.float32))
    if T.cuda.is_available():
        grid_offset = grid_offset.cuda()
        anchor_t = anchor_t.cuda()

    # T.sigmoid(conv_raw_dxdy)的shape是(N, n, n, 3, 2)，grid_offset的shape是(n, n, 3, 2)。属于不同shape相加
    pred_xy = (T.sigmoid(conv_raw_dxdy) + grid_offset) * stride
    pred_wh = (T.exp(conv_raw_dwdh) * anchor_t) * stride
    pred_xywh = T.cat((pred_xy, pred_wh), dim=-1)

    pred_conf = T.sigmoid(conv_raw_conf)
    pred_prob = T.sigmoid(conv_raw_prob)
    return T.cat((pred_xywh, pred_conf, pred_prob), dim=-1)

def yolo_loss(args, num_classes, iou_loss_thresh, anchors, alpha_1, alpha_2, alpha_3):
    conv_lbbox = args[0]   # (?, ?, ?, 3, num_classes+5)
    conv_mbbox = args[1]   # (?, ?, ?, 3, num_classes+5)
    conv_sbbox = args[2]   # (?, ?, ?, 3, num_classes+5)
    label_sbbox = args[3]   # (?, ?, ?, 3, num_classes+5)
    label_mbbox = args[4]   # (?, ?, ?, 3, num_classes+5)
    label_lbbox = args[5]   # (?, ?, ?, 3, num_classes+5)
    true_sbboxes = args[6]   # (?, 150, 4)
    true_mbboxes = args[7]   # (?, 150, 4)
    true_lbboxes = args[8]   # (?, 150, 4)
    pred_sbbox = decode(conv_sbbox, anchors[0], 8)
    pred_mbbox = decode(conv_mbbox, anchors[1], 16)
    pred_lbbox = decode(conv_lbbox, anchors[2], 32)
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbboxes, 8, num_classes, iou_loss_thresh, alpha=alpha_1)
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbboxes, 16, num_classes, iou_loss_thresh, alpha=alpha_2)
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbboxes, 32, num_classes, iou_loss_thresh, alpha=alpha_3)
    return loss_sbbox + loss_mbbox + loss_lbbox

class YoloLoss(torch.nn.Module):
    def __init__(self, num_classes, iou_loss_thresh, anchors, alpha_1, alpha_2, alpha_3):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss_thresh = iou_loss_thresh
        self.anchors = anchors
        self.alpha_1 = alpha_1    # 小感受野输出层的focal_loss的alpha
        self.alpha_2 = alpha_2    # 中感受野输出层的focal_loss的alpha
        self.alpha_3 = alpha_3    # 大感受野输出层的focal_loss的alpha

    def forward(self, args):
        return yolo_loss(args, self.num_classes, self.iou_loss_thresh, self.anchors, self.alpha_1, self.alpha_2, self.alpha_3)

class Conv2dUnit(torch.nn.Module):
    def __init__(self, input_dim, filters, kernels, stride, padding):
        super(Conv2dUnit, self).__init__()
        self.conv = torch.nn.Conv2d(input_dim, filters, kernel_size=kernels, stride=stride, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(filters)
        self.leakyreLU = torch.nn.LeakyReLU(0.1)

        # 参数初始化。不这么初始化，容易梯度爆炸nan
        self.conv.weight.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, input_dim, kernels[0], kernels[1])))
        self.bn.weight.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
        self.bn.bias.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
        self.bn.running_mean.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
        self.bn.running_var.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyreLU(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dUnit(input_dim, filters, (1, 1), stride=1, padding=0)
        self.conv2 = Conv2dUnit(filters, 2*filters, (3, 3), stride=1, padding=1)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x

class StackResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, filters, n):
        super(StackResidualBlock, self).__init__()
        self.sequential = torch.nn.Sequential()
        for i in range(n):
            self.sequential.add_module('stack_%d' % (i+1,), ResidualBlock(input_dim, filters))
    def forward(self, x):
        for residual_block in self.sequential:
            x = residual_block(x)
        return x

class Darknet(torch.nn.Module):
    def __init__(self, num_classes, initial_filters=32):
        super(Darknet, self).__init__()
        self.num_classes = num_classes
        i32 = initial_filters
        i64 = i32 * 2
        i128 = i32 * 4
        i256 = i32 * 8
        i512 = i32 * 16
        i1024 = i32 * 32

        ''' darknet53部分，这里所有卷积层都没有偏移bias=False '''
        self.conv1 = Conv2dUnit(3, i32, (3, 3), stride=1, padding=1)
        self.conv2 = Conv2dUnit(i32, i64, (3, 3), stride=2, padding=1)
        self.stack_residual_block_1 = StackResidualBlock(i64, i32, n=1)

        self.conv3 = Conv2dUnit(i64, i128, (3, 3), stride=2, padding=1)
        self.stack_residual_block_2 = StackResidualBlock(i128, i64, n=2)

        self.conv4 = Conv2dUnit(i128, i256, (3, 3), stride=2, padding=1)
        self.stack_residual_block_3 = StackResidualBlock(i256, i128, n=8)

        self.conv5 = Conv2dUnit(i256, i512, (3, 3), stride=2, padding=1)
        self.stack_residual_block_4 = StackResidualBlock(i512, i256, n=8)

        self.conv6 = Conv2dUnit(i512, i1024, (3, 3), stride=2, padding=1)
        self.stack_residual_block_5 = StackResidualBlock(i1024, i512, n=4)
        ''' darknet53部分结束 '''

        self.conv53 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        self.conv54 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        self.conv55 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        self.conv56 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        self.conv57 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)

        self.conv58 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        self.conv59 = torch.nn.Conv2d(i1024, 3*(num_classes + 5), kernel_size=(1, 1))

        self.conv60 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.conv61 = Conv2dUnit(i256+i512, i256, (1, 1), stride=1, padding=0)
        self.conv62 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        self.conv63 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        self.conv64 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        self.conv65 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)

        self.conv66 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        self.conv67 = torch.nn.Conv2d(i512, 3*(num_classes + 5), kernel_size=(1, 1))

        self.conv68 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.conv69 = Conv2dUnit(i128+i256, i128, (1, 1), stride=1, padding=0)
        self.conv70 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
        self.conv71 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.conv72 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
        self.conv73 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.conv74 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)

        self.conv75 = torch.nn.Conv2d(i256, 3*(num_classes + 5), kernel_size=(1, 1))

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.stack_residual_block_1(x)
        x = self.conv3(x)
        x = self.stack_residual_block_2(x)
        x = self.conv4(x)
        act11 = self.stack_residual_block_3(x)
        x = self.conv5(act11)
        act19 = self.stack_residual_block_4(x)
        x = self.conv6(act19)
        act23 = self.stack_residual_block_5(x)

        x = self.conv53(act23)
        x = self.conv54(x)
        x = self.conv55(x)
        x = self.conv56(x)
        lkrelu57 = self.conv57(x)

        x = self.conv58(lkrelu57)
        y1 = self.conv59(x)
        y1 = y1.view(y1.size(0), 3, (self.num_classes + 5), y1.size(2), y1.size(3))  # reshape

        x = self.conv60(lkrelu57)
        x = self.upsample1(x)
        x = torch.cat((x, act19), dim=1)

        x = self.conv61(x)
        x = self.conv62(x)
        x = self.conv63(x)
        x = self.conv64(x)
        lkrelu64 = self.conv65(x)

        x = self.conv66(lkrelu64)
        y2 = self.conv67(x)
        y2 = y2.view(y2.size(0), 3, (self.num_classes + 5), y2.size(2), y2.size(3))  # reshape

        x = self.conv68(lkrelu64)
        x = self.upsample2(x)
        x = torch.cat((x, act11), dim=1)

        x = self.conv69(x)
        x = self.conv70(x)
        x = self.conv71(x)
        x = self.conv72(x)
        x = self.conv73(x)
        x = self.conv74(x)
        y3 = self.conv75(x)
        y3 = y3.view(y3.size(0), 3, (self.num_classes + 5), y3.size(2), y3.size(3))  # reshape

        # 相当于numpy的transpose()，交换下标
        y1 = y1.permute(0, 3, 4, 1, 2)
        y2 = y2.permute(0, 3, 4, 1, 2)
        y3 = y3.permute(0, 3, 4, 1, 2)
        return y1, y2, y3
