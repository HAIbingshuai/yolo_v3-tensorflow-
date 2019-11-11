# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license     ：MIT License
#   Author      : haibingshuai 
#   Created date: 2019/10/29 15:18
#   Description :
# ================================================================
import shutil
import tensorflow as tf
import os
from core.model_yolo_v3 import model_yolo
from core import utils
from core.model_config import config
from core.Dataset import Dataset
import numpy as np
import time
from tqdm import tqdm


class Train_yolo_v3(object):
    def __init__(self):
        # model_need
        self.train_tpye = config.YOLO.TRAIN_type
        self.anchor_per_scale = config.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(config.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.logdir = config.YOLO.LOGDIR

        # train_need
        self.trainset = Dataset('train')
        self.steps_per_period = len(self.trainset)

        self.testset = Dataset('test')
        self.initial_weight = config.TRAIN.INITIAL_WEIGHT

        self.learn_rate_initial = config.TRAIN.RATE_INITIAL
        self.learn_rate_end = config.TRAIN.RATE_END
        self.first_stage_epochs = config.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = config.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods = config.TRAIN.WARMUP_EPOCHS

        # common_need
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

        with tf.name_scope('define_input'):
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')

            self.label_smt_box = tf.placeholder(dtype=tf.float32, name='label_smart_box')
            self.label_mid_box = tf.placeholder(dtype=tf.float32, name='label_middle_box')
            self.label_big_box = tf.placeholder(dtype=tf.float32, name='label_big_box')

            self.true_smt_boxes = tf.placeholder(dtype=tf.float32, name='true_smart_boxes')
            self.true_mid_boxes = tf.placeholder(dtype=tf.float32, name='true_middle_boxes')
            self.true_big_boxes = tf.placeholder(dtype=tf.float32, name='true_big_boxes')

            self.label_three_box = [self.label_smt_box, self.label_mid_box, self.label_big_box]
            self.true_three_boxs = [self.true_smt_boxes, self.true_mid_boxes, self.true_big_boxes]

        with tf.name_scope("define_loss"):
            self.model = model_yolo(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.loss_giou, self.loss_confidence, self.loss_classify = self.model.compute_loss(
                self.label_three_box, self.true_three_boxs
            )
            self.loss = self.loss_giou + self.loss_confidence + self.loss_classify

        with tf.name_scope('learn_rate'):
            # warmup stage:
            # 刚开始训练的时候是非常不稳定的，因此刚开始的学习率应当设置得很低很低，来保证网络能够具有良好的收敛性。
            # 但是较低的学习率会使得训练过程变得非常缓慢，因此这里会采用以较低学习率逐渐增大至较高学习率的方式实现网络训练的“热身”阶段，
            # consine decay stage:
            # 但是如果我们使得网络训练的 loss 最小，那么一直使用较高学习率是不合适的，因为它会使得权重的梯度一直来回震荡.
            # 很难使训练的损失值达到全局最低谷。
            # 因此最后采用了这篇论文里的 cosine 的衰减方式。

            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')

            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period, dtype=tf.float64,
                                       name='warmup_steps')

            two_epochs_sum = (self.first_stage_epochs + self.second_stage_epochs)
            train_steps = tf.constant(two_epochs_sum * self.steps_per_period, dtype=tf.float64, name='train_steps')

            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_initial,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_initial - self.learn_rate_end) *
                                 (1 + tf.cos((self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            self.global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_first_stage_train"):
            val_all_name = []
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                val_all_name.append(var_name)
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_smt_BOX', 'conv_mid_BOX', 'conv_big_BOX']:
                    self.first_stage_trainable_var_list.append(var)
            self.first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate). \
                minimize(self.loss, var_list=self.first_stage_trainable_var_list, global_step=self.global_step_update)

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            self.second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate). \
                minimize(self.loss, var_list=second_stage_trainable_var_list, global_step=self.global_step_update)

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss", self.loss_giou)
            tf.summary.scalar("confidence_loss", self.loss_confidence)
            tf.summary.scalar("classify_loss", self.loss_classify)
            tf.summary.scalar("total_loss", self.loss)

            if os.path.exists(self.logdir):
                shutil.rmtree(self.logdir)
            os.mkdir(self.logdir)

            self.write_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.logdir, graph=self.sess.graph)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            if self.train_tpye == '接着上一步继续训练':
                print('=> Restoring weights from: ./checkpoint/')
                model_file = tf.train.latest_checkpoint('./checkpoint/')
                self.saver.restore(self.sess, model_file)
                print('已经加载最新model参数（来自/checkpoint/）-------------ok')
                fir_step_num = self.first_stage_epochs
            elif self.train_tpye == '加载预训练 新开训练':
                print('=> Restoring weights from: %s ... ' % self.initial_weight)
                self.loader.restore(self.sess, self.initial_weight)
                fir_step_num = 1
                print('已经成功加载预训练模型参数（来自' + self.initial_weight + '）-------------ok')
            else:
                fir_step_num = 1
        except:
            print('=> %s 不存在!!!\n开始从头裸训练' % self.initial_weight)
            fir_step_num = 1

        for epoch in range(fir_step_num, 1 + self.first_stage_epochs + self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.first_stage_optimizer  # self.train_op_with_frozen_variables
            else:
                train_op = self.second_stage_optimizer  # self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                        self.input_data: train_data[0],
                        self.label_smt_box: train_data[1],
                        self.label_mid_box: train_data[2],
                        self.label_big_box: train_data[3],
                        self.true_smt_boxes: train_data[4],
                        self.true_mid_boxes: train_data[5],
                        self.true_big_boxes: train_data[6],
                        self.trainable: True,
                    })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" % train_step_loss)

            for test_data in self.testset:
                test_step_loss = self.sess.run(self.loss, feed_dict={
                    self.input_data: test_data[0],
                    self.label_smt_box: test_data[1],
                    self.label_mid_box: test_data[2],
                    self.label_big_box: test_data[3],
                    self.true_smt_boxes: test_data[4],
                    self.true_mid_boxes: test_data[5],
                    self.true_big_boxes: test_data[6],
                    self.trainable: False,
                })
                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))

            self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == '__main__': Train_yolo_v3().train()
