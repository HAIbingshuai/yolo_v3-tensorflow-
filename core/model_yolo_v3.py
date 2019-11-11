import numpy as np
import tensorflow as tf
from core import utils, common, darknet53
from core.model_config import config
from core.utils import focal,bbox_giou


class model_yolo(object):
    def __init__(self, input_data, trainable):
        self.trainable = trainable

        self.classes = utils.read_class_names(config.YOLO.CLASSES)
        self.class_num = len(self.classes)

        self.strides = np.array(config.YOLO.STRIDES)  # [8,16,32]     (3,)
        self.anchors = utils.get_anchors(config.YOLO.ANCHORS)  # (3,3,2)
        self.anchor_per_scale = config.YOLO.ANCHOR_PER_SCALE  # 3
        self.iou_loss_thresh = config.YOLO.IOU_LOSS_THRESH  # 阈值
        self.upsample_method = config.YOLO.UPSAMPLE_METHOD  # 上采样的方法

        try:
            self.conv_smt_box, self.conv_mid_box, self.conv_big_box = self.yolo_v3_network(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3_network!")

        with tf.variable_scope('pred_smt_box'):
            #   shape = (52,52, 3 * (self.class_num + 5)   #[3,2]  #8
            self.pred_smt_box = self.decode(self.conv_smt_box, self.anchors[0], self.strides[0])
        with tf.variable_scope('pred_mid_box'):
            #   shape = (26, 26, 3 * (self.class_num + 5)  #[3,2]  #16
            self.pred_mid_box = self.decode(self.conv_mid_box, self.anchors[1], self.strides[1])
        with tf.variable_scope('pred_big_box'):
            #   shape = (13, 13, 3 * (self.class_num + 5)  #[3,2]  #32
            self.pred_big_box = self.decode(self.conv_big_box, self.anchors[2], self.strides[2])

    # 五次DBL [A]                                                                   和DBL+conv(no_bn/激活的DBL)
    # 继续上层[A]+DBL+上采样+【上层middle_data_2】concat+五次DBL [B]                    和DBL+conv(no_bn/激活的DBL)
    # 继续上层[B]+DBL+上采样+【上层middle_data_1】concat+五次DBL                        和DBL+conv(no_bn/激活的DBL)
    def yolo_v3_network(self, input_layer):
        # 输入层进入 Darknet-53 网络后，得到了三个分支
        middle_data_1, middle_data_2, conv = darknet53.darknet53(input_layer)

        # 五次DBL(卷积，BN[批量规范化],激活函数)
        conv = common.convolutional_Layer(conv, (1, 1, 1024, 512), name='conv52_five_1')
        conv = common.convolutional_Layer(conv, (3, 3, 512, 1024), name='conv53_five_1')
        conv = common.convolutional_Layer(conv, (1, 1, 1024, 512), name='conv54_five_1')
        conv = common.convolutional_Layer(conv, (3, 3, 512, 1024), name='conv55_five_1')
        conv = common.convolutional_Layer(conv, (1, 1, 1024, 512), name='conv56_five_1')

        # 第一尺度  ----------->conv_BIG_BOX 用于预测大尺寸物体，shape = [None, 13, 13, 255]
        conv_first_obj_branch = common.convolutional_Layer(conv, (3, 3, 512, 1024), name='conv_big_obj_branch')
        conv_big_BOX = common.convolutional_Layer(conv_first_obj_branch, (1, 1, 1024, 3 * (self.class_num + 5)),
                                                  activate=False,
                                                  bn=False, name='conv_big_BOX')

        # --------------------------------
        conv = common.convolutional_Layer(conv, (1, 1, 512, 256), name='conv57')
        conv = common.upsample(conv, 'middle_box_upsample')
        conv = tf.concat([conv, middle_data_2], axis=-1)  # axis,拼接方向，1y轴，2x轴，3...-1-->最后一维度
        # 此刻的输入维度（512+256)
        conv = common.convolutional_Layer(conv, (1, 1, 768, 256), name='conv58_five_2')
        conv = common.convolutional_Layer(conv, (3, 3, 256, 512), name='conv59_five_2')
        conv = common.convolutional_Layer(conv, (1, 1, 512, 256), name='conv60_five_2')
        conv = common.convolutional_Layer(conv, (3, 3, 256, 512), name='conv61_five_2')
        conv = common.convolutional_Layer(conv, (1, 1, 512, 256), name='conv62_five_2')

        # 第二尺度  ----------->conv_middle_BOX 用于预测中等尺寸物体，shape = [None, 26, 26, 255]
        conv_second_obj_branch = common.convolutional_Layer(conv, (3, 3, 256, 512), name='conv_mid_obj_branch')
        conv_middle_BOX = common.convolutional_Layer(conv_second_obj_branch, (1, 1, 512, 3 * (self.class_num + 5)),
                                                     activate=False,
                                                     bn=False, name='conv_mid_BOX')

        # --------------------------------
        conv = common.convolutional_Layer(conv, (1, 1, 256, 128), name='conv63')
        conv = common.upsample(conv, 'smart_box_upsample')
        conv = tf.concat([conv, middle_data_1], axis=-1)
        # 此刻的输入维度（256+128）
        conv = common.convolutional_Layer(conv, (1, 1, 384, 128), name='conv64_five_3')
        conv = common.convolutional_Layer(conv, (3, 3, 128, 256), name='conv65_five_3')
        conv = common.convolutional_Layer(conv, (1, 1, 256, 128), name='conv66_five_3')
        conv = common.convolutional_Layer(conv, (3, 3, 128, 256), name='conv67_five_3')
        conv = common.convolutional_Layer(conv, (1, 1, 256, 128), name='conv68_five_3')

        # 第三尺度  ----------->conv_smart_BOX 用于预测小尺寸物体，shape = [None, 52, 52, 255]
        conv_third_obj_branch = common.convolutional_Layer(conv, (3, 3, 128, 256), name='conv_smt_obj_branch')
        conv_smart_BOX = common.convolutional_Layer(conv_third_obj_branch, (1, 1, 256, 3 * (self.class_num + 5)),
                                                    activate=False,
                                                    bn=False, name='conv_smt_BOX')

        return conv_smart_BOX, conv_middle_BOX, conv_big_BOX

    def decode(self, conv_output, anchors, stride):

        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """
        # conv_output, anchors, stride:
        #   shape = (batch_size,52,52, 3 * (self.class_num + 5)  #[3,2]  #8
        #   shape = (batch_size,26,26, 3 * (self.class_num + 5)  #[3,2]  #16
        #   shape = (batch_size,13,13, 3 * (self.class_num + 5)  #[3,2]  #32

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, anchor_per_scale, 5 + self.class_num))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_confidence = conv_output[:, :, :, :, 4:5]
        conv_raw_classify = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_confidence = tf.sigmoid(conv_raw_confidence)
        pred_classify = tf.sigmoid(conv_raw_classify)

        return tf.concat([pred_xywh, pred_confidence, pred_classify], axis=-1)

    def compute_loss(self, lab_three_box, true_three_box):
        [label_smt_box, label_mid_box, label_big_box] = lab_three_box
        [true_smt_box, true_mid_box, true_big_box] = true_three_box
        with tf.name_scope('smaller_box_loss'):
            loss_smt_box = self.loss_layer(self.conv_smt_box, self.pred_smt_box, label_smt_box, true_smt_box,
                                           stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mid_box = self.loss_layer(self.conv_mid_box, self.pred_mid_box, label_mid_box, true_mid_box,
                                           stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_big_box = self.loss_layer(self.conv_big_box, self.pred_big_box, label_big_box, true_big_box,
                                           stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_smt_box[0] + loss_mid_box[0] + loss_big_box[0]

        with tf.name_scope('conf_loss'):
            confidence_loss = loss_smt_box[1] + loss_mid_box[1] + loss_big_box[1]

        with tf.name_scope('prob_loss'):
            classify_loss = loss_smt_box[2] + loss_mid_box[2] + loss_big_box[2]

        return giou_loss, confidence_loss, classify_loss

    # 核输出，核转化坐标lab，lab
    def loss_layer(self, conv, pred, label, bboxes, stride):

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size, self.anchor_per_scale, 5 + self.class_num))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]
        # respond_bbox的意思是如果网格单元中包含物体，那么就会计算边界框损失；
        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)
        # 边界框的尺寸越小，（此样本的误差权重应该大，弥补小尺寸iou和大尺寸iou的不平衡），bbox_loss_scale的值就越大。v1用的是边长开根号，这是为了弱化边界框尺寸对损失值的影响；
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        # 两个边界框之间的GIoU值越大，giou的损失值就会越小, 因此网络会朝着预测框与真实框重叠度较高的方向去优化。
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        iou = bbox_giou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = focal(respond_bbox, pred_conf)

        confidence_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )
        confidence_loss = tf.reduce_mean(tf.reduce_sum(confidence_loss, axis=[1, 2, 3, 4]))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        classify_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
        classify_loss = tf.reduce_mean(tf.reduce_sum(classify_loss, axis=[1, 2, 3, 4]))

        return giou_loss, confidence_loss, classify_loss
