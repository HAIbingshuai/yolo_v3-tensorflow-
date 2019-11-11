from core import common
import tensorflow as tf


# DBL+RES1/2/8/8/4+FC_layer(全连接层此时为池化+全连接+softmax，但yolo并没有用)
def darknet53(input_data):  # 用了五次下采样，所以图片的大小应该维持32的倍数
    with tf.variable_scope('darknet53'):
        # 0
        input_data = common.convolutional_Layer(input_data, filters_shape=(3, 3, 3, 32), name='conv0')

        # -------------
        # 1
        input_data = common.convolutional_Layer(input_data, filters_shape=(3, 3, 32, 64), name='conv1', downsample=True)
        # 2-3(1*2)
        for i in range(1):
            input_data = common.residual_block(input_data, 64, 32, 64, name='residual%d' % (i + 0))

        # -------------
        # 4
        input_data = common.convolutional_Layer(input_data, filters_shape=(3, 3, 64, 128), downsample=True,
                                                name='conv4')
        # 5-8(2*2)
        for i in range(2):
            input_data = common.residual_block(input_data, 128, 64, 128, name='residual%d' % (i + 1))

        # -------------
        # 9
        input_data = common.convolutional_Layer(input_data, filters_shape=(3, 3, 128, 256), downsample=True,
                                                name='conv9')
        # 10-25(8*2)
        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, name='residual%d' % (i + 3))

        middle_data_1 = input_data

        # -------------
        # 26
        input_data = common.convolutional_Layer(input_data, filters_shape=(3, 3, 256, 512), downsample=True,
                                                name='conv26')
        # 27-42(8*2)
        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, name='residual%d' % (i + 11))

        middle_data_2 = input_data

        # -------------
        # 43
        input_data = common.convolutional_Layer(input_data, filters_shape=(3, 3, 512, 1024), downsample=True,
                                                name='conv43')
        # 44-51(4*2)
        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, name='residual%d' % (i + 19))
        # 备注 darknet中还有三层（Avgpool、Connected 和 softmax layer）
        # 用于在 Imagenet 数据集上作分类训练用的，yolo是另一种思路，是不会用到这三层的，只是用前面来特征提取，谁让darknet53这么给力捏。

    return middle_data_1, middle_data_2, input_data
