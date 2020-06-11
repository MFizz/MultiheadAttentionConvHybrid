import log.log_utils
import os
import models.words2vec
import models.vid2sentence
import constants.constants as const
import data.creation.vid2sentence
import logging
import tensorflow as tf
import sql_models.sql_control

from sql_models.input_data_models import SentenceSQL, LabelSQL, Word2VecSQL, Video2SentenceDataSQL, \
    Video2SentenceModelSQL, Video2SentenceTrainBlockSQL, PredictionResults, PredictionExample

if __name__ == '__main__':
    sql_models.sql_control.get_sql_sentences()
    sql_models.sql_control.get_sql_labels(const.SOMETHING_ALL_LABELS_FILENAME)

    tf.logging.set_verbosity(logging.INFO)
    log.log_utils.add_main_logging()
    w2v_model = models.words2vec.Words2vec('something_something_with_start_token', const.SOMETHING_LABELS_FILENAME,
                                           embedding_size=128, neg_samples=64)
    sql_models.sql_control.get_sql_word2vec_model(w2v_model)

    proj_name = 'something_something_jpg'
    v2s_dataset = data.creation.vid2sentence.V2SDataset(
        proj_name, w2v_model,
        const.SOMETHING_VIDEO_BASE_PATH,
        train_labels_filename=const.SOMETHING_TRAIN_LABELS_FILENAME,
        eval_labels_filename=const.SOMETHING_EVAL_LABELS_FILENAME)

    # v2s_dataset = data.creation.vid2sentence.V2SOpticalFlowDataset(
    #     proj_name, w2v_model,
    #     const.SOMETHING_VIDEO_BASE_PATH,
    #     train_labels_filename=const.SOMETHING_TRAIN_LABELS_FILENAME,
    #     eval_labels_filename=const.SOMETHING_EVAL_LABELS_FILENAME)

    sql_models.sql_control.get_sql_vid2sen_data(v2s_dataset)

    v2s_test_dataset = data.creation.vid2sentence.V2SDataset(proj_name, w2v_model,
                                                             const.SOMETHING_VIDEO_BASE_PATH,
                                                             prediction_labels_filename=const.SOMETHING_TEST_LABELS_FILENAME,
                                                             prediction_name='test')
    sql_models.sql_control.get_sql_vid2sen_data(v2s_test_dataset)
    v2s_model = models.vid2sentence.Vid2SentenceTPU(proj_name, [0.0002], v2s_dataset, const.EncoderType.MultiHeadAttention, video_emb_type=const.VideoEmbeddingType.MobilenetV1, bfloat16_mode=False, optimizer_type=const.OptimizerType.Adam, warm_start_dir=const.MOBILE_1_128_PATH)
    sql_models.sql_control.get_sql_vid2sen_model(v2s_model)
    start_steps, end_steps, learning_rates = v2s_model.early_stopping_train_eval_loop(use_tpu=True)
    # start_steps, end_steps, learning_rates = v2s_model.fixed_epoch_train_eval_loop(use_tpu=False, train_batch_size=2,
    #                                                                                 eval_batch_size=2,
    #                                                                                 train_examples_per_epoch=2,
    #                                                                                 eval_examples_per_epoch=2,
    #                                                                                 epochs=2)
    sql_models.sql_control.create_sql_train_blocks(v2s_model, start_steps, end_steps, learning_rates, 1, 1)
    blocks = zip(start_steps, end_steps, learning_rates)
    beam_size = 2
    for start_step, end_step, learning_rate in blocks:
        batch_pred_sens, batch_logits, batch_softmax, batch_video_ids = \
            v2s_model.predict(v2s_test_dataset,
                              beam_size,
                              os.path.join(v2s_model.model_dir,
                                           'MultiHeadAttention_encoder/MobilenetV1_video_emb/lr_{}'.format(str(learning_rate))),
                              with_labels=False, predict_batch_size=1)
        sql_pred_res = sql_models.sql_control.create_prediction_result(v2s_model, end_step, learning_rate, v2s_test_dataset, 3)
        batched_pred = zip(batch_pred_sens, batch_logits, batch_softmax, batch_video_ids)
        for prediction in batched_pred:
            beam_sentences, logits, softmaxes, video_id = prediction
            examples = zip(beam_sentences, logits, softmaxes)
            for sen, logit, softmax in examples:
                sql_models.sql_control.create_prediction_example(sql_pred_res, logit, softmax, sen, video_id)
