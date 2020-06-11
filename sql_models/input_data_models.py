import peewee as pw
import constants.constants as const

something_database = pw.SqliteDatabase(None)


def initialize(db=const.MODEL_DB):
    something_database.init(db, pragmas={'journal_mode': 'wal',
                                          'cache_size': 10000,  # 10000 pages, or ~40MB
                                          'synchronous': 0})
    return something_database


class Results(pw.Model):
    name = pw.CharField()
    birthday = pw.DateField()

    class Meta:
        database = something_database


class SentenceSQL(pw.Model):
    sentence = pw.CharField(unique=True, primary_key=True)

    class Meta:
        database = something_database


class LabelSQL(pw.Model):
    label_name = pw.CharField()
    data_set_name = pw.CharField()
    video_id = pw.IntegerField()
    sentence = pw.ForeignKeyField(SentenceSQL, backref='labels', null=True)

    class Meta:
        database = something_database


class Word2VecSQL(pw.Model):
    data_set_name = pw.CharField()
    data_set_file = pw.CharField()
    embedding_size = pw.IntegerField()
    neg_samples = pw.IntegerField()
    skip_word_window = pw.IntegerField()
    batch_size = pw.IntegerField()
    epochs = pw.IntegerField()
    eval_frq = pw.IntegerField()
    model_dir = pw.CharField()

    class Meta:
        database = something_database


class Video2SentenceDataSQL(pw.Model):
    name = pw.CharField()
    w2v_model = pw.ForeignKeyField(Word2VecSQL, backref='v2s')
    input_frames_path = pw.CharField()
    train_labels_filename = pw.CharField(null=True)
    eval_labels_filename = pw.CharField(null=True)
    predict_labels_filename = pw.CharField(null=True)
    predict_name = pw.CharField(null=True)
    model_dir = pw.CharField()
    sentence_max = pw.IntegerField()
    frames_max = pw.IntegerField()
    max_height = pw.IntegerField()
    max_width = pw.IntegerField()
    preprocessed = pw.BooleanField()
    jpg_quality = pw.IntegerField()
    optical_flow = pw.BooleanField()
    optical_flow_bound = pw.IntegerField(null=True)
    optical_flow_resize_factor = pw.IntegerField(null=True)
    jpg_skip = pw.IntegerField(null=True),
    opt_flow_bracket = pw.IntegerField(null=True)

    class Meta:
        database = something_database


class Video2SentenceModelSQL(pw.Model):
    name = pw.CharField()
    dataset = pw.ForeignKeyField(Word2VecSQL, backref='video2sentences')
    model_fn = pw.CharField()
    encoder_type = pw.CharField()
    normalization_fn = pw.CharField()
    activation = pw.CharField()
    video_emb_type = pw.CharField()
    model_dir = pw.CharField()
    frame_shape = pw.CharField()
    batch_size = pw.IntegerField()
    epochs_per_eval = pw.IntegerField()
    dropout_rate = pw.FloatField()
    layer_stack = pw.IntegerField()
    mult_heads = pw.IntegerField()
    ff_hidden_units = pw.IntegerField()
    bfloat16_mode = pw.BooleanField()
    optimizer_type = pw.CharField()
    adam_beta1 = pw.FloatField(null=True)
    adam_beta2 = pw.FloatField(null=True)
    adam_epsilon = pw.FloatField(null=True)
    adafactor_decay_rate = pw.FloatField(null=True)
    adafactor_beta1 = pw.FloatField(null=True)
    adafactor_epsilon1 = pw.FloatField(null=True)
    adafactor_epsilon2 = pw.FloatField(null=True)
    label_smoothing = pw.FloatField(null=True)
    max_n_gram = pw.IntegerField(null=True)
    delta_step = pw.IntegerField(null=True)
    min_step = pw.IntegerField(null=True)
    delta_value = pw.IntegerField(null=True)
    warm_start_dir = pw.CharField(null=True)

    class Meta:
        database = something_database


class VideoId2Sentence(pw.Model):
    video_id = pw.IntegerField()
    sentence = pw.CharField()

    class Meta:
        database = something_database


class Video2SentenceTrainBlockSQL(pw.Model):
    start_step = pw.IntegerField()
    end_step = pw.IntegerField()
    batch_size = pw.IntegerField()
    examples_per_epoch = pw.CharField()
    learning_rate = pw.CharField()
    model = pw.ForeignKeyField(Video2SentenceModelSQL, backref='video2sentencetrains')

    class Meta:
        database = something_database


class PredictionResults(pw.Model):
    current_step = pw.IntegerField()
    learning_rate = pw.FloatField()
    test_dataset = pw.ForeignKeyField(Video2SentenceDataSQL, backref='predictionresults')
    beam_size = pw.IntegerField()
    model = pw.ForeignKeyField(Video2SentenceModelSQL, backref='predictionresults')

    class Meta:
        database = something_database


class PredictionExample(pw.Model):
    prediction_result = pw.ForeignKeyField(PredictionResults, backref='predicitonexamples')
    logit = pw.FloatField()
    softmax = pw.FloatField()
    pred_sentence = pw.CharField()
    real_sentence = pw.CharField(null=True)
    video_id = pw.IntegerField()

    class Meta:
        database = something_database


