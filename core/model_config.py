from easydict import EasyDict as edict

config = edict()
config.YOLO = edict()
config.TRAIN = edict()
config.TEST = edict()

# model
config.YOLO.CLASSES = './data/anchors_and_classes/class.names'
config.YOLO.ANCHORS = './data/anchors_and_classes/basline_anchors.txt'
config.YOLO.STRIDES = [8, 16, 32]
config.YOLO.ANCHOR_PER_SCALE = 3
config.YOLO.IOU_LOSS_THRESH = 0.5
config.YOLO.UPSAMPLE_METHOD = 'resize'
config.YOLO.LOGDIR = './checkpoint/log/'

# ----------------
# '接着上一步继续训练'、'不加载预训练 新开训练'、'加载预训练 新开训练'
config.YOLO.TRAIN_type = '接着上一步继续训练'  #

# DATAS
## trian
config.TRAIN.FISRT_STAGE_EPOCHS = 1  # 冻结darknet前，训练之后网络
config.TRAIN.SECOND_STAGE_EPOCHS = 4  # 全训练
config.TRAIN.AUGMENTION = False
config.TRAIN.INPUT_SIZE = 608  # 320, 352, 384, 416, 448, 480, 512, 544,
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.DATAS_PATH = './data/dataset/train.txt'
config.TRAIN.INITIAL_WEIGHT = './checkpoint/initial_weight/model.ckpt'
config.TRAIN.WARMUP_EPOCHS = 2
config.TRAIN.RATE_INITIAL = 1e-4
config.TRAIN.RATE_END = 1e-6

## test
config.TEST.Augmentation = False
config.TEST.INPUT_SIZE = 608
config.TEST.BATCH_SIZE = 4
config.TEST.DATAS_PATH = './data/dataset/test.txt'
config.TEST.WRITE_IMAGE = True
config.TEST.WRITE_IMAGE_PATH = "./data/detection/"
config.TEST.WRITE_IMAGE_SHOW_LABEL = False
config.TEST.SHOW_LABEL = False
config.TEST.WEIGHT_FILE = "./checkpoint/yolov3_test_loss=?.ckpt-5"
config.TEST.SCORE_THRESHOLD = 0.3
config.TEST.IOU_THRESHOLD = 0.45
