# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license     : MIT License
#   Author      : HAIbingshuai 
#   Created date: 2019/11/11 14:53
#   Description :
# ================================================================

import tensorflow as tf

from core.model_yolo_v3 import model_yolo

pb_file = "./susong_header_5k_model_1.pb"
ckpt_file = "./checkpoint/yolov3_test_loss=0.4528.ckpt-2"
output_node_names = ["input/input_data", "pred_smt_box/concat_2", "pred_mid_box/concat_2", "pred_big_box/concat_2"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, name='input_data')

model = model_yolo(input_data, trainable=False)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def=sess.graph.as_graph_def(),
                                                                   output_node_names=output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
