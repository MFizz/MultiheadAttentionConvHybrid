import os
import enum

GCLOUD = False

GS_PATH = 'gs://mitch-conv-attention'
HD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if GCLOUD:
    BASE_PATH = GS_PATH
else:
    BASE_PATH = HD_PATH

# - logging paths
MAIN_LOGS = os.path.join(HD_PATH, 'log', 'main_logs')
DEFAULT_MAIN_CONF_FILE = os.path.join(HD_PATH, 'log', 'main_logs_conf.yaml')
DEFAULT_WORDS2VEC_CONF_FILE = os.path.join(HD_PATH, 'log', 'words2vec_logs_conf.yaml')
DEFAULT_VID2SENTENCE_CONF_FILE = os.path.join(HD_PATH, 'log', 'vid2sentence_logs_conf.yaml')
DEFAULT_LR_CONF_FILE = os.path.join(HD_PATH, 'log', 'lr_logs_conf.yaml')

# - word2vec paths
WORD2VEC_DEFAULT_MODEL_PATH = os.path.join(BASE_PATH, 'data', 'sets', 'word2vec')

# - vid2sentence paths
VID2SENTENCE_DEFAULT_MODEL_PATH = os.path.join(BASE_PATH, 'data', 'sets', 'vid2sentence')

# - model paths
# - - warm start dirs
INC_4_PATH = os.path.join(BASE_PATH, 'data', 'sets', 'inception_v4', 'inception_v4.ckpt')
MOBILE_1_128_PATH = os.path.join(BASE_PATH, 'data', 'sets', 'mobilenet_v1', 'mobile128', 'mobilenet_v1_0.25_128.ckpt')

# - data set specific paths
# - - something something data set
SOMETHING_DATA_BASE_PATH = os.path.join(HD_PATH, 'data', 'sets', 'something_something')
SOMETHING_LABELS_FILENAME = os.path.join(SOMETHING_DATA_BASE_PATH, 'labels',
                                         'something-something-v1-labels.csv')
SOMETHING_TRAIN_LABELS_FILENAME = os.path.join(SOMETHING_DATA_BASE_PATH, 'labels',
                                               'something-something-v1-train.csv')
SOMETHING_EVAL_LABELS_FILENAME = os.path.join(SOMETHING_DATA_BASE_PATH, 'labels',
                                              'something-something-v1-validation.csv')
SOMETHING_TEST_LABELS_FILENAME = os.path.join(SOMETHING_DATA_BASE_PATH, 'labels',
                                              'something-something-v1-test.csv')
SOMETHING_ALL_LABELS_FILENAME = os.path.join(SOMETHING_DATA_BASE_PATH, 'labels',
                                              'something-something-v1-all.csv')
SOMETHING_VIDEO_BASE_PATH = '/home/mitch/20bn-datasets/videos/'
SOMETHING_OPT_FLOW_BASE_PATH = '/home/mitch/20bn-datasets/opt_flow/'

# sql paths
MODEL_DB = os.path.join(HD_PATH, 'sql_models', 'sql_on_old.db')
DATA_SET_NAME = 'something-something'



class InputDataTypes(enum.Enum):
    VideoJpg = 1
    PreprocessedVideoJpg = 2
    OpticalFlowVideoJpg = 3


class EncoderType(enum.Enum):
    Convolution = 1
    MultiHeadAttention = 2


class VideoEmbeddingType(enum.Enum):
    SpatialCnn = 1
    InceptionV4 = 2
    MobilenetV1 = 3
    MobilenetV2 = 4

class WeightMode(enum.Enum):
    Linear = 1
    Gauss = 2
    Identity = 3

class OptimizerType(enum.Enum):
    Adam = 1
    Adafactor = 2

