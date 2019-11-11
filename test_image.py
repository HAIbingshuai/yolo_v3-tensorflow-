# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license='MIT License'
#   Author      : haibingshuai 
#   Created date: 2019/10/29 18:05
#   Description :
# ================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import os

return_elements = ["input/input_data:0", "pred_smt_box/concat_2:0", "pred_mid_box/concat_2:0",
                   "pred_big_box/concat_2:0"]
pb_file = "./susong_header_5k_model_1.pb"
image_path_ = "./data_test/in"
image_out_path_ = "./data_test/out"
num_classes = 1
input_size = 608  # 416
graph = tf.Graph()

pic_list_path = [[os.path.join(image_path_, one), os.path.join(image_out_path_, one)] for one in
                 os.listdir(image_path_)]

with tf.Session(graph=graph) as sess:
    for image_path, image_out_path in pic_list_path:
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = utils.image_pretreat_process(np.copy(original_image), input_size)
        image_data = image_data[np.newaxis, ...]

        return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
            feed_dict={return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.2)
        bboxes = utils.nms(bboxes, 0.5, method='nms')
        print(len(bboxes))
        image = utils.draw_bbox(original_image, bboxes)

        image = Image.fromarray(image)
        image.show()
        image = np.array(image)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_out_path, image)
