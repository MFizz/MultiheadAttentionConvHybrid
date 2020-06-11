from math import log
from numpy import array
import tensorflow as tf
import models.vid2sentence_modules


def decode_fn(ids, voc_size):
    ids_shape = ids.get_shape().as_list()
    return tf.random_uniform((ids_shape[0], ids_shape[1], voc_size))


def compute_batch_indices(batch_size, beam_size):
  """Computes the i'th coordinate that contains the batch index for gathers.

  Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
  batch the beam item is in. This will create the i of the i,j coordinate
  needed for the gather.

  Args:
    batch_size: Batch size
    beam_size: Size of the beam.
  Returns:
    batch_pos: [batch_size, beam_size] tensor of ids
  """
  batch_pos = tf.range(batch_size * beam_size) // beam_size
  batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
  return batch_pos


if __name__ == '__main__':
    voc_size = 15
    beam_size = 5
    alpha = 0.6
    max_sentence = 20
    emb_lookup = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14]]
    batch_size = 1
    decode = lambda x: decode_fn(x, voc_size)
    word2id = {'unk': 0, '<s>': 1, '</s>': 2, 'unk3': 3, 'unk4': 4, 'unk5': 5, 'unk6': 6, 'unk7': 7, 'unk8': 8, 'unk9': 9, 'unk10': 10, 'unk11': 11, 'unk12': 12, 'unk13': 13, 'unk14': 14}
    topk_finished_seqs, topk_finished_log_probs, topsoft = models.vid2sentence_modules.beam_search_decoder(beam_size, batch_size, decode, max_sentence, emb_lookup, word2id, voc_size, alpha)

    with tf.Session() as sess:
        print(sess.run([topk_finished_seqs, topk_finished_log_probs, topsoft]))