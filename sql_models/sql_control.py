import sql_models.input_data_models
import os
import models.words2vec
import data.creation.vid2sentence
from data.creation import word_processing
from peewee import chunked
from sql_models.input_data_models import SentenceSQL, LabelSQL, Word2VecSQL, Video2SentenceDataSQL, \
    Video2SentenceModelSQL, Video2SentenceTrainBlockSQL, PredictionResults, PredictionExample
import constants.constants as const
import pandas as pd
import constants.hyper_params as hp


def get_sql_sentences(sentence_file=const.SOMETHING_LABELS_FILENAME, database=None):
    db = _get_db(database)
    with db:
        db.create_tables([SentenceSQL])
        query = SentenceSQL.select()
        if not query.exists(db):
            query = create_sentences(db, sentence_file)
        sentences = [row.sentence for row in query]
        sentences_df = pd.DataFrame(sentences, columns=['sentence'])
    return sentences_df


def create_sentences(db, sentence_file=const.SOMETHING_LABELS_FILENAME):
    sentences_list = word_processing.read_csv_label_sentences(sentence_file)
    sentence_dicts = [{'sentence': sentence} for sentence in sentences_list]
    with db.atomic():
        for batch in chunked(sentence_dicts, 100):
            SentenceSQL.insert_many(batch).execute()
    return SentenceSQL.select()


def get_sql_labels(label_file, database=None):
    db = _get_db(database)
    with db:
        db.create_tables([LabelSQL, SentenceSQL])
        if not SentenceSQL.select().exists(db):
            create_sentences(db)
        label_name = os.path.basename(label_file).split('.')[0]
        query = LabelSQL.select().where(LabelSQL.label_name == label_name)
        if not query.exists(db):
            query = create_labels(db, label_file)
        labels = [(row.label_name, row.data_set_name, row.video_id, row.sentence_id) for row in query]
        labels_df = pd.DataFrame(labels, columns=['labels_name', 'data_set_name', 'video_id', 'sentence'])
    return labels_df


def create_labels(database, label_file, data_set_name=const.DATA_SET_NAME):
    label_df = pd.read_csv(label_file, delimiter=';', header=None)
    if label_df.shape[1] == 1:
        label_df.columns = ['video_id']
        label_df['sentence'] = None
    else:
        label_df.columns = ['video_id', 'sentence']
    label_name = os.path.basename(label_file).split('.')[0]
    label_dicts = [{'label_name': label_name,
                    'data_set_name': data_set_name,
                    'video_id': row.video_id,
                    'sentence_id': word_processing.norm_sentence(row.sentence)}
                   for _, row in label_df.iterrows()]
    pd.set_option('display.max_colwidth', -1)
    with database.atomic():
        for batch in chunked(label_dicts, 100):
            LabelSQL.insert_many(batch).execute()
    return LabelSQL.select().where(LabelSQL.label_name == label_name)


def get_sql_word2vec_model(word2vec, database=None, initialized=False):
    if not initialized:
        db = _get_db(database)
        with db:
            w2v = _get_sql_word2vec_model_helper(word2vec, db)
    else:
        w2v = _get_sql_word2vec_model_helper(word2vec, database)
    return w2v


def _get_sql_word2vec_model_helper(word2vec, db):
    db.create_tables([Word2VecSQL])
    w2v, _ = Word2VecSQL.get_or_create(data_set_name=word2vec.name,
                                       data_set_file=word2vec.input_filename,
                                       embedding_size=word2vec.embedding_size,
                                       neg_samples=word2vec.neg_samples,
                                       skip_word_window=word2vec.skip_word_window,
                                       batch_size=word2vec.batch_size,
                                       epochs=word2vec.epochs,
                                       eval_frq=word2vec.eval_frq,
                                       model_dir=word2vec.model_directory)
    return w2v


def get_sql_vid2sen_data(v2s_data, database=None, initialized=False):
    if initialized:
        return _get_sql_vid2sen_data_helper(v2s_data, database)
    else:
        db = _get_db(database)
        with db:
            return _get_sql_vid2sen_data_helper(v2s_data, db)


def _get_sql_vid2sen_data_helper(v2s_data, db):
    db.create_tables([Video2SentenceDataSQL])
    v2s, _ = Video2SentenceDataSQL.get_or_create(name=v2s_data.name,
                                                 w2v_model=get_sql_word2vec_model(v2s_data.w2v_model, database=db,
                                                                                  initialized=True),
                                                 input_frames_path=v2s_data.frames_path,
                                                 train_labels_filename=v2s_data.train_labels_filename,
                                                 eval_labels_filename=v2s_data.eval_labels_filename,
                                                 predict_labels_filename=v2s_data.predict_labels_filename,
                                                 predict_name=v2s_data.predict_name,
                                                 model_dir=v2s_data.model_dir,
                                                 sentence_max=v2s_data.sentence_max,
                                                 frames_max=v2s_data.frames_max,
                                                 max_height=v2s_data.max_height,
                                                 max_width=v2s_data.max_width,
                                                 preprocessed=isinstance(v2s_data,
                                                                         data.creation.vid2sentence.V2SprecDataset),
                                                 jpg_quality=hp.JPG_QUAL,
                                                 optical_flow=isinstance(v2s_data,
                                                                         data.creation.vid2sentence.V2SOpticalFlowDataset),
                                                 optical_flow_bound=hp.BOUND,
                                                 optical_flow_resize_factor=hp.FLOW_RESIZE_FACTOR,
                                                 jpg_skip=hp.JPG_SKIP,
                                                 opt_flow_bracket=hp.BRACKET
                                                 )
    return v2s


def get_sql_vid2sen_model(v2s_model, database=None, initialized=False):
    if initialized:
        return _get_sql_vid2sen_model_helper(v2s_model, database)
    else:
        db = _get_db(database)
        with db:
            return _get_sql_vid2sen_model_helper(v2s_model, db)


def _get_sql_vid2sen_model_helper(v2s_model, db):
    db.create_tables([Video2SentenceModelSQL])
    v2s, _ = Video2SentenceModelSQL.get_or_create(name=v2s_model.name,
                                                  dataset=get_sql_vid2sen_data(v2s_model.dataset, db, initialized=True),
                                                  model_fn=v2s_model.model_fn.__name__,
                                                  encoder_type=v2s_model.enc_type,
                                                  normalization_fn=v2s_model.normalization_fn.__name__,
                                                  activation=v2s_model.activation.__name__,
                                                  video_emb_type=v2s_model.video_emb_type,
                                                  model_dir=v2s_model.model_dir,
                                                  frame_shape=str(v2s_model.frame_shape),
                                                  batch_size=v2s_model.batch_size,
                                                  epochs_per_eval=v2s_model.epochs_per_eval,
                                                  dropout_rate=v2s_model.dropout_rate,
                                                  layer_stack=v2s_model.layer_stack,
                                                  mult_heads=v2s_model.mult_heads,
                                                  ff_hidden_units=v2s_model.ff_hidden_units,
                                                  bfloat16_mode=v2s_model.bfloat16_mode,
                                                  optimizer_type=v2s_model.optimizer_type,
                                                  adam_beta1=v2s_model.adam_beta1,
                                                  adam_beta2=v2s_model.adam_beta2,
                                                  adam_epsilon=v2s_model.adam_epsilon,
                                                  adafactor_decay_rate=v2s_model.adafactor_decay_rate,
                                                  adafactor_beta1=v2s_model.adafactor_beta1,
                                                  adafactor_epsilon1=v2s_model.adafactor_epsilon1,
                                                  adafactor_epsilon2=v2s_model.adafactor_epsilon2,
                                                  label_smoothing=v2s_model.label_smoothing,
                                                  max_n_gram=v2s_model.max_n_gram,
                                                  delta_step=v2s_model.delta_step,
                                                  min_step=v2s_model.min_step,
                                                  delta_value=v2s_model.delta_value,
                                                  warm_start_dir=v2s_model.warm_start_dir)
    return v2s


def create_sql_train_blocks(v2s_model, start_steps, end_steps, learning_rates, batch_size, examples_per_epoch, database=None):
    blocks = zip(start_steps, end_steps, learning_rates)
    db = _get_db(database)
    with db:
        model = get_sql_vid2sen_model(v2s_model, database=db, initialized=True)
        db.create_tables([Video2SentenceTrainBlockSQL])
        for start_step, end_step, learning_rate in blocks:
            sql_block, _ = Video2SentenceTrainBlockSQL.get_or_create(start_step=start_step,
                                                                     end_step=end_step,
                                                                     batch_size=batch_size,
                                                                     examples_per_epoch=examples_per_epoch,
                                                                     learning_rate=learning_rate,
                                                                     model=model)


def create_prediction_result(v2s_model, current_step, learning_rate, test_dataset, beam_size, database=None):
    db = _get_db(database)
    with db:
        db.create_tables([PredictionResults])
        sql_pred_res, _ = PredictionResults.get_or_create(current_step=current_step,
                                                          learning_rate=learning_rate,
                                                          test_dataset=get_sql_vid2sen_data(test_dataset, db, initialized=True),
                                                          beam_size=beam_size,
                                                          model=get_sql_vid2sen_model(v2s_model, db, initialized=True))
        return sql_pred_res


def create_prediction_example(pred_result, logit, softmax, pred_sentence, video_id, real_sentence=None, database=None):
    db = _get_db(database)
    with db:
        db.create_tables([PredictionExample])
        sql_pred_ex, _ = PredictionExample.get_or_create(prediction_result=pred_result,
                                                         logit=logit,
                                                         softmax=softmax,
                                                         pred_sentence=" ".join(pred_sentence),
                                                         video_id=video_id,
                                                         real_sentence=real_sentence)
        return sql_pred_ex



def _get_db(db):
    if db:
        return sql_models.input_data_models.initialize(db)
    else:
        return sql_models.input_data_models.initialize()


if __name__ == '__main__':
    # print(get_sentences())
    w2v_model = models.words2vec.Words2vec('something_something_with_start_token', const.SOMETHING_LABELS_FILENAME,
                                           embedding_size=128, neg_samples=64)
    print(get_sql_word2vec_model(w2v_model))
