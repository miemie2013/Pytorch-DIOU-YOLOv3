# -*- coding: utf-8 -*-

import torch
import random
import colorsys
import cv2
import time
import os
import numpy as np

from model.darknet_yolo_pt import Darknet


class Decode(object):
    def __init__(self, obj_threshold, nms_threshold, input_shape, model_path, file_path, initial_filters):
        self._t1 = obj_threshold
        self._t2 = nms_threshold
        self.input_shape = input_shape
        self.all_classes = self.get_classes(file_path)
        self.num_classes = len(self.all_classes)
        self._yolo = Darknet(self.num_classes, initial_filters=initial_filters)
        if torch.cuda.is_available():  # 如果有gpu可用，模型（包括了权重weight）存放在gpu显存里
            self._yolo = self._yolo.cuda()
        self._yolo.load_state_dict(torch.load(model_path))
        self._yolo.eval()   # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式. 不这样做的化会产生不一致的推理结果.

    # 处理一张图片
    def detect_image(self, image):
        pimage = self.process_image(image)

        start = time.time()
        boxes, scores, classes = self.predict(pimage, image.shape)
        print('time: {0:.6f}s'.format(time.time() - start))
        if boxes is not None:
            self.draw(image, boxes, scores, classes)
        return image

    # 处理视频
    def detect_video(self, video):
        video_path = os.path.join("videos", "test", video)
        camera = cv2.VideoCapture(video_path)
        cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

        # Prepare for saving the detected video
        sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mpeg')

        vout = cv2.VideoWriter()
        vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

        while True:
            res, frame = camera.read()

            if not res:
                break

            image = self.detect_image(frame)
            cv2.imshow("detection", image)

            # Save the video frame by frame
            vout.write(image)

            if cv2.waitKey(110) & 0xff == 27:
                break

        vout.release()
        camera.release()

    def get_classes(self, file):
        with open(file) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]

        return class_names

    def draw(self, image, boxes, scores, classes):
        image_h, image_w, _ = image.shape
        # 定义颜色
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for box, score, cl in zip(boxes, scores, classes):
            x0, y0, x1, y1 = box
            left = max(0, np.floor(x0 + 0.5).astype(int))
            top = max(0, np.floor(y0 + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
            bbox_color = colors[cl]
            bbox_thick = 1 if min(image_h, image_w) < 400 else 2
            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
            bbox_mess = '%s: %.2f' % (self.all_classes[cl], score)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        #     print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        #     print('box coordinate x,y,w,h: {0}'.format(box))
        # print()

    def training_transform(self, height, width, output_height, output_width):
        height_scale, width_scale = output_height / height, output_width / width
        scale = min(height_scale, width_scale)
        resize_height, resize_width = round(height * scale), round(width * scale)
        pad_top = (output_height - resize_height) // 2
        pad_left = (output_width - resize_width) // 2
        A = np.float32([[scale, 0.0], [0.0, scale]])
        B = np.float32([[pad_left], [pad_top]])
        M = np.hstack([A, B])
        return M, output_height, output_width

    def process_image(self, img):
        h, w = img.shape[:2]
        M, h_out, w_out = self.training_transform(h, w, self.input_shape[0], self.input_shape[1])
        # 填充黑边缩放
        letterbox = cv2.warpAffine(img, M, (w_out, h_out))
        pimage = np.float32(letterbox) / 255.
        pimage = np.expand_dims(pimage, axis=0)
        return pimage

    def predict(self, image, shape):
        image = image.transpose(0, 3, 1, 2)
        image = torch.Tensor(image)

        start = time.time()
        outs = self._yolo(image)
        print('\ndarknet time: {0:.6f}s'.format(time.time() - start))

        # win10下这一步很耗时
        outs = [o.cpu().detach().numpy() for o in outs]
        # pytorch后处理
        start = time.time()
        boxes, scores, classes = self._yolo_out(outs, shape, self.input_shape)
        print('post process time: {0:.6f}s'.format(time.time() - start))

        return boxes, scores, classes


    def _process_feats(self, out, anchors, input_shape, use_cuda):
        txty = out[..., :2]
        twth = out[..., 2:4]
        conf = torch.sigmoid(out[..., 4:5])
        class_p = torch.sigmoid(out[..., 5:])
        scores = conf*class_p

        batch, grid_r, grid_c, box_num, cat = scores.size()

        # 取前256个，可能有nms中被消除的框
        # topk_scores, topk_inds = torch.topk(scores.view(-1, ), min(256, batch*grid_r*grid_c*box_num*cat))
        topk_scores, topk_inds = torch.topk(scores.view(-1, ), 256)

        topk_inds2 = topk_inds % (box_num * cat)
        box_id = (topk_inds2 / cat).int()
        class_id = (topk_inds2 % cat).int()

        topk_inds3 = (topk_inds / (box_num * cat)).int()
        topk_ys = (topk_inds3 / grid_c).int().float()
        topk_xs = (topk_inds3 % grid_c).int().float()
        topk_xys = torch.cat([topk_xs.unsqueeze(-1), topk_ys.unsqueeze(-1)], dim=1)

        pos_ys = topk_inds3 * box_num * 2 + 2 * box_id + 1
        pos_xs = topk_inds3 * box_num * 2 + 2 * box_id

        txty_flat = txty.contiguous().view(-1, )
        twth_flat = twth.contiguous().view(-1, )
        tys = txty_flat.gather(0, pos_ys.long())
        txs = txty_flat.gather(0, pos_xs.long())
        ths = twth_flat.gather(0, pos_ys.long())
        tws = twth_flat.gather(0, pos_xs.long())

        txty_ = torch.cat([txs.unsqueeze(-1), tys.unsqueeze(-1)], dim=1)
        twth_ = torch.cat([tws.unsqueeze(-1), ths.unsqueeze(-1)], dim=1)

        anchors_h = anchors.gather(0, (box_id * 2 + 1).long())
        anchors_w = anchors.gather(0, (box_id * 2).long())
        anchors_wh = torch.cat([anchors_w.unsqueeze(-1), anchors_h.unsqueeze(-1)], dim=1)

        if use_cuda:
            bxby = (topk_xys + torch.sigmoid(txty_)) * input_shape / torch.Tensor([grid_r, grid_c]).cuda()
        else:
            bxby = (topk_xys + torch.sigmoid(txty_)) * input_shape / torch.Tensor([grid_r, grid_c])
        bwbh = anchors_wh * torch.exp(twth_)
        x0y0 = bxby - bwbh/2
        x1y1 = bxby + bwbh/2
        box = torch.cat([x0y0, x1y1], dim=1)
        return box, topk_scores, class_id

    def _nms_boxes(self, boxes, scores):
        x0 = boxes[:, 0]
        y0 = boxes[:, 1]
        x1 = boxes[:, 2]
        y1 = boxes[:, 3]

        areas = (x1-x0) * (y1-y0)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x0[i], x0[order[1:]])
            yy1 = np.maximum(y0[i], y0[order[1:]])
            xx2 = np.minimum(x1[i], x1[order[1:]])
            yy2 = np.minimum(y1[i], y1[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self._t2)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep


    def _yolo_out(self, outs, shape, input_shape):
        h, w = input_shape
        input_shape = torch.Tensor(input_shape)
        anchors = torch.Tensor([[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]])
        use_cuda = torch.cuda.is_available()
        # use_cuda = False
        if use_cuda:
            input_shape = input_shape.cuda()
            anchors = anchors.cuda()

        boxes, scores, classes = [], [], []
        i = 0

        start = time.time()
        for out in outs:
            out = torch.Tensor(out)
            if use_cuda:
                out = out.cuda()
            boxes_per_field, scores_per_field, classes_per_field = self._process_feats(out, anchors[i], input_shape, use_cuda)
            if use_cuda:
                boxes_per_field, scores_per_field, classes_per_field = boxes_per_field.cpu().numpy(), scores_per_field.cpu().numpy(), classes_per_field.cpu().numpy()
            else:
                boxes_per_field, scores_per_field, classes_per_field = boxes_per_field.numpy(), scores_per_field.numpy(), classes_per_field.numpy()

            pos = np.where(scores_per_field >= self._t1)
            if len(pos) != 0:
                boxes_per_field = boxes_per_field[pos]
                scores_per_field = scores_per_field[pos]
                classes_per_field = classes_per_field[pos]
                boxes.append(boxes_per_field)
                scores.append(scores_per_field)
                classes.append(classes_per_field)
            i += 1

        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)
        classes = np.concatenate(classes)

        print('feat time: {0:.6f}s'.format(time.time() - start))
        start = time.time()

        # Scale boxes back to original image shape.
        iw, ih = shape[1], shape[0]
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) / (2*scale)
        dy = (h - nh) / (2*scale)
        sc = max(iw, ih)
        sc_y, sc_x = sc/h, sc/w
        image_dims = [sc_x, sc_y, sc_x, sc_y]
        dd = [dx, dy, dx, dy]
        boxes = boxes * image_dims - dd

        nboxes, nscores, nclasses = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            s = scores[inds]
            c = classes[inds]

            keep = self._nms_boxes(b, s)

            nboxes.append(b[keep])
            nscores.append(s[keep])
            nclasses.append(c[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        scores = np.concatenate(nscores)
        classes = np.concatenate(nclasses)
        print('nms time: {0:.6f}s'.format(time.time() - start))

        return boxes, scores, classes


