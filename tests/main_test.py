import os
import log.log_utils
import models.words2vec
import models.vid2sentence
import constants.constants as const
import data.creation.vid2sentence

if __name__ == '__main__':
    log.log_utils.add_main_logging()
    w2v_model = models.words2vec.Words2vec('something_something-test', const.SOMETHING_LABELS_FILENAME,
                                           embedding_size=64, neg_samples=32)

    # v2s_name = 'something_something-convmult-test-jpg'
    # v2s_dataset = data.creation.vid2sentence.V2SDataset(v2s_name, w2v_model,
    #                                                      const.SOMETHING_TRAIN_LABELS_FILENAME,
    #                                                      const.SOMETHING_EVAL_LABELS_FILENAME,
    #                                                      const.SOMETHING_VIDEO_BASE_PATH)
    # v2s_model = models.vid2sentence.Vid2Sentence(v2s_name, [0.001], v2s_dataset, const.EncoderType.Convolution, video_emb_type=const.VideoEmbeddingType.InceptionV4)

    # v2s_name = 'something_something-convmult-test-prec-jpg'
    # v2s_dataset = data.creation.vid2sentence.V2SprecDataset(v2s_name, w2v_model,
    #                                                         const.SOMETHING_TRAIN_LABELS_FILENAME,
    #                                                         const.SOMETHING_EVAL_LABELS_FILENAME,
    #                                                         const.SOMETHING_VIDEO_BASE_PATH)
    #
    # v2s_model = models.vid2sentence.Vid2Sentence(v2s_name, [0.001], v2s_dataset,
    #                                              const.EncoderType.Convolution)

    v2s_name = 'something_something-convmult-test-opt-jpg'
    v2s_dataset = data.creation.vid2sentence.V2SOpticalFlowDataset(v2s_name, w2v_model,
                                                               const.SOMETHING_TRAIN_LABELS_FILENAME,
                                                               const.SOMETHING_EVAL_LABELS_FILENAME,
                                                               const.SOMETHING_VIDEO_BASE_PATH)

    v2s_model = models.vid2sentence.Vid2Sentence(v2s_name, [0.001], v2s_dataset,
                                                 const.EncoderType.MultiHeadAttention)
    v2s_model.early_stopping_train_eval_loop()
    # v2s_model.predict(const.SOMETHING_TEST_LABELS_FILENAME,
    #                   os.path.join(v2s_model.model_dir, 'MultiHeadAttention_encoder', 'bs_2_lr_0.001'))