import log.log_utils
import models.words2vec
import models.vid2sentence
import constants.constants as const
import data.creation.vid2sentence

if __name__ == '__main__':
    log.log_utils.add_main_logging()
    w2v_model = models.words2vec.Words2vec('something_something-test', const.SOMETHING_LABELS_FILENAME,
                                           embedding_size=64, neg_samples=32)

    v2s_name = 'something_something-convmult-test'
    v2s_dataset = data.creation.vid2sentence.V2SprecJpgDataset(v2s_name, w2v_model,
                                                         const.SOMETHING_TRAIN_LABELS_FILENAME,
                                                         const.SOMETHING_EVAL_LABELS_FILENAME,
                                                         const.SOMETHING_VIDEO_BASE_PATH)
    v2s_model = models.vid2sentence.Vid2SentenceConvMultPreprocJpg(v2s_name, [0.0001], v2s_dataset)
