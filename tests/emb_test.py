import tensorflow as tf
import numpy as np
from numpy.ma.testutils import assert_equal

import models.words2vec
import constants.constants as const

if __name__ == '__main__':
    w2v_model = models.words2vec.Words2vec('something_something_val_test', const.SOMETHING_LABELS_FILENAME,
                                           embedding_size=128, neg_samples=64)
    input = [1, 2, 3, 4, 5, 6]
    t_input = tf.convert_to_tensor(np.array(input))
    labels = tf.map_fn(lambda x: tf.nn.embedding_lookup(tf.convert_to_tensor(w2v_model.emb_lookup_table, tf.float32), x),
                       t_input, dtype=tf.float32)
    embedded_sentence = w2v_model.emb_lookup_table[input]

    with tf.Session() as sess:
        t_embs = sess.run(labels)
        assert_equal(embedded_sentence, t_embs)