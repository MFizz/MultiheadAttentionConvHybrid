import tensorflow as tf
import constants.constants as const

# words2vec
W2V_EMBEDDING_SIZE = 128
W2V_NEG_SAMPLES = 64
W2V_SKIP_WORD_WINDOW = 1
W2V_BATCH_SIZE = 100
W2V_EPOCHS = 500
W2V_EVAL_FRQ = 2


# vid2sentence
# data set params
V2S_SENTENCE_MAX = 20
V2S_FRAMES_MAX = 26
V2S_MAX_HEIGHT = 128
V2S_MAX_WIDTH = 128
V2S_TFR_EX_NUM = 10000
FRAME_SHAPE = (V2S_MAX_HEIGHT, V2S_MAX_WIDTH, 3)
# optional, will be set automaticallyQ
CNN_MODEL_SHAPE = (6, 6, 1536)
V2S_BFLOAT16_MODE = False

# hyper params
V2S_BATCH_SIZE = 2
V2S_EPOCHS_PER_EVAL = 1
V2S_DROPOUT_RATE = 0.1
V2S_LAYERS = 6
V2S_NUM_MULT_HEADS = 8
V2S_FF_HIDDEN_UNITS = 1024
V2S_ACTIVATION = tf.nn.selu

# optimizer params
V2S_LBL_SMOOTHING = 0.1
V2S_WEIGHT_MODE = const.WeightMode.Gauss
V2S_OPTIMIZER_TYPE = const.OptimizerType.Adam

# optimizer params - adam
V2S_ADAM_BETA1 = 0.9
V2S_ADAM_BETA2 = 0.999
V2S_ADAM_EPSILON = 1e-08

# optimizer params - adafactor
V2S_ADAFACTOR_DECAY = None
V2S_ADAFACTOR_BETA = 0.0
V2S_ADAFACTOR_EPSILON1 = 1e-30
V2S_ADAFACTOR_EPSILON2 = 1e-3

# eval metric params
V2S_MAX_N_GRAM = 4

# early stopping params
V2S_DELTA_VAL = 0.0001
V2S_DELTA_STEP = 1300
V2S_MIN_STEP = 0

# tpu params
# Number of training steps to run on the Cloud TPU before returning control.
TPU_ITERATIONS = 100
# A single Cloud TPU has 8 shards.
TPU_NUM_SHARDS = 8
TPU_ZONE = ""
TPU_GCP_NAME = ""
TPU_NAME = ""
TPU_LOG_STEP = 1
TPU_TRAIN_EXAMPLES_PER_EPOCH = 80000
TPU_EVAL_EXAMPLES_PER_EPOCH = 6912
TPU_TRAIN_BATCH_SIZE = 128  # 16
TPU_EVAL_BATCH_SIZE = 128
TPU_PREDICT_BATCH_SIZE = 128
TPU_EPOCHS = 2

# Jpg encoding
JPG_SKIP = 10
JPG_QUAL = 60

# data pipeline
PREFETCH_ELEMENTS = 1
PARSER_PARALLEL_CALLS = 8

# checkpoints
SAVE_CHK_STEP = 500
KEEP_CHK = 10

# optical flow
BRACKET = 5
SKIP = 1
BOUND = 15
FLOW_RESIZE_FACTOR = 2  # needs to be 2^x

# beam search
ALPHA = 0.6
