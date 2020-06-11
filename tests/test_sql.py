import pytest
import sql_models.sql_control
import constants.constants as const
import constants.hyper_params as hp
from unittest import mock


def test_get_sentences():
    sentences = sql_models.sql_control.get_sql_sentences(database=':memory:')
    assert sentences.shape[0] == 174, "wrong number of label sentences"
    assert sentences.at[0, 'sentence'] == "<s> holding something </s>", "wrong order of label sentences"
    assert sentences.at[173, 'sentence'] == "<s> poking a hole into some substance </s>", "wrong order of label sentences"


# @pytest.mark.slow add slow testing
def test_get_labels():
    labels1 = sql_models.sql_control.get_sql_labels(const.SOMETHING_TEST_LABELS_FILENAME, database=':memory:')
    assert labels1.shape[0] == 10960, "wrong number of labels"
    assert labels1.at[0, 'sentence'] is None
    assert labels1.at[0, 'video_id'] == 105723
    assert labels1.at[10959, 'video_id'] == 105050
    assert labels1.at[0, 'labels_name'] == "something-something-v1-test"
    assert labels1.at[0, 'data_set_name'] == "something-something"

    labels1 = sql_models.sql_control.get_sql_labels(const.SOMETHING_TRAIN_LABELS_FILENAME, database=':memory:')
    assert labels1.shape[0] == 86017, "wrong number of labels"
    assert labels1.at[0, 'sentence'] == "<s> something falling like a feather or paper </s>"
    assert labels1.at[0, 'video_id'] == 100218
    assert labels1.at[86016, 'sentence'] == "<s> plugging something into something </s>"
    assert labels1.at[86016, 'video_id'] == 62090
    assert labels1.at[0, 'labels_name'] == "something-something-v1-train"
    assert labels1.at[0, 'data_set_name'] == "something-something"


def test_sql_model():
    w2v_model = mock.Mock()
    w2v_model.data_set_name = 'test_words_name'
    w2v_model.data_set_file = '/test_word_name_csv'
    w2v_model.embedding_size = hp.W2V_EMBEDDING_SIZE
    w2v_model.neg_samples = hp.W2V_NEG_SAMPLES
    w2v_model.skip_word_window = hp.W2V_SKIP_WORD_WINDOW
    w2v_model.batch_size = hp.W2V_BATCH_SIZE
    w2v_model.epoch = hp.W2V_EPOCHS
    w2v_model.eval_frq = hp.W2V_EVAL_FRQ
    w2v_model.model_dir = const.WORD2VEC_DEFAULT_MODEL_PATH

    sql_w2v = sql_models.sql_control.get_sql_word2vec_model(w2v_model)






