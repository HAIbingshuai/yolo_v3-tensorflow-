import cv2
import os
import random
import numpy as np
import tensorflow as tf
from core import utils
from core.model_config import config


class Dataset(object):
    def __init__(self, dataset_type):
        self.input_size = config.TRAIN.INPUT_SIZE if dataset_type == 'train' else config.TEST.INPUT_SIZE
        self.batch_size = config.TRAIN.BATCH_SIZE if dataset_type == 'train' else config.TEST.BATCH_SIZE
        self.datas_path = config.TRAIN.DATAS_PATH if dataset_type == 'train' else config.TEST.DATAS_PATH
        self.data_aug = config.TRAIN.AUGMENTION if dataset_type == 'train' else config.TEST.Augmentation

        self.classes = utils.read_class_names(config.YOLO.CLASSES)
        self.num_classes = len(self.classes)

        self.datas = self.load_datas()
        self.total_num_datas = len(self.datas)
        self.batch_count = 0
        self.num_batchs = int(np.ceil(self.total_num_datas / self.batch_size))

        self.anchors = np.array(utils.get_anchors(config.YOLO.ANCHORS))

        self.train_input_size = self.input_size

        self.strides = np.array(config.YOLO.STRIDES)

        self.anchor_per_scale = config.YOLO.ANCHOR_PER_SCALE  # 即长宽比例（1:1,1:2,2:1）

        self.max_bbox_per_scale = 100  # config.YOLO.MAX_BBOX_PER_SCALE  #

    def __len__(self):
        return self.num_batchs

    def __iter__(self):
        return self

    def load_datas(self):
        with open(self.datas_path, 'r') as f:
            txt = f.readlines()
            datas = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        random.shuffle(datas)
        return datas

    def parse_data(self, data):
        line = data.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s 介个路径没有东西，检查检查 " % image_path)

        image = cv2.imread(image_path)
        image = np.array(image)
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        if self.data_aug:  # 水平翻转，裁剪（0.5以内），缩放
            # 噪声点，污点等操作
            image, bboxes = utils.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = utils.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = utils.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_pretreat_process(np.copy(image),
                                                     self.train_input_size, np.copy(bboxes))

        return image, bboxes

    def __next__(self):

        with tf.device('/cpu:0'):
            # 输入尺寸416--> 三个尺寸缩放（缩小）8/16/32倍,--52/26/13(粗中细)
            self.train_output_sizes = self.train_input_size // self.strides

            # 2*[416*416*3]
            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

            # 2*52*52*3*(5+n)/2*26*26*3*(5+n)/2*13*13*3*(5+n)
            batch_label_smt_box = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                            self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mid_box = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                            self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lag_box = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                            self.anchor_per_scale, 5 + self.num_classes))
            # 2*150*4
            batch_smt_boxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mid_boxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lag_boxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            # 假如    num_batchs = 100次
            #           batch_size = 2个
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.total_num_datas:
                        index -= self.total_num_datas

                    data = self.datas[index]
                    image, bboxes = self.parse_data(data)
                    label_smt_box, label_mid_box, label_lag_box, smt_boxes, mid_boxes, lag_boxes = self.preprocess_true_boxes(
                        bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_smt_box[num, :, :, :, :] = label_smt_box
                    batch_label_mid_box[num, :, :, :, :] = label_mid_box
                    batch_label_lag_box[num, :, :, :, :] = label_lag_box
                    batch_smt_boxes[num, :, :] = smt_boxes
                    batch_mid_boxes[num, :, :] = mid_boxes
                    batch_lag_boxes[num, :, :] = lag_boxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_smt_box, batch_label_mid_box, batch_label_lag_box, \
                       batch_smt_boxes, batch_mid_boxes, batch_lag_boxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.datas)
                raise StopIteration

    def preprocess_true_boxes(self, bboxes):
        # define
        label = [np.zeros(
            (self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale, 5 + self.num_classes)) for i
            in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]

        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_position = bbox[:4]
            bbox_class_ind = bbox[4]

            # classify_value
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            # 中点，各个尺度
            bbox_xywh = np.concatenate(
                [(bbox_position[2:] + bbox_position[:2]) * 0.5, bbox_position[2:] - bbox_position[:2]], axis=-1)
            bbox_three_scaled_xywh = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            #
            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_three_scaled_xywh[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = utils.bbox_iou(bbox_three_scaled_xywh[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_three_scaled_xywh[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_three_scaled_xywh[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
