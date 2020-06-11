import tensorflow as tf
import collections
import numpy as np
from collections import OrderedDict


def get_accuracy(labels, predictions, mask=None, name='accuracy'):
    with tf.variable_scope(name):
        correct = tf.cast(tf.equal(tf.to_int32(predictions), labels), dtype=tf.float32)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            correct *= mask
            num = tf.reduce_sum(mask)
        else:
            num = tf.cast(tf.size(labels), tf.float32)

        correct = tf.reduce_sum(correct)

        total = tf.get_local_variable('total', [], dtype=tf.float32, initializer=tf.zeros_initializer())
        count = tf.get_local_variable('count', [], dtype=tf.float32, initializer=tf.zeros_initializer())

        update_total_op = tf.assign_add(total, correct)
        update_count_op = tf.assign_add(count, num)

        mean_t = _safe_div(total, count, 'value')
        update_op = _safe_div(update_total_op, update_count_op, 'update_op')

        return mean_t, update_op


def _safe_div(numerator, denominator, name):
    t = tf.truediv(numerator, denominator)
    zero = tf.zeros_like(t, dtype=denominator.dtype)
    condition = tf.greater(denominator, zero)
    zero = tf.cast(zero, t.dtype)

    return tf.where(condition, t, zero, name=name)


def bleu_score(predictions, labels, max_n):

    numerators_cum = tf.get_local_variable('numerators', [max_n], dtype=tf.float32, initializer=tf.zeros_initializer())
    denominators_cum = tf.get_local_variable('denominators', [max_n], dtype=tf.float32, initializer=tf.zeros_initializer())
    cand_lengths_cum = tf.get_local_variable('cand_lengths', [], dtype=tf.float32, initializer=tf.zeros_initializer())
    ref_lengths_cum = tf.get_local_variable('ref_lengths', [], dtype=tf.float32, initializer=tf.zeros_initializer())
    num, den, can, ref = tf.py_func(_get_bleu_score_params, (predictions, labels, max_n), (tf.float32, tf.float32, tf.float32, tf.float32))

    update_num_op = tf.assign_add(numerators_cum, num)
    update_denom_op = tf.assign_add(denominators_cum, den)
    update_cand_op = tf.assign_add(cand_lengths_cum, can)
    update_ref_op = tf.assign_add(ref_lengths_cum, ref)

    bleu = _calc_bleu_score(numerators_cum, denominators_cum, cand_lengths_cum, ref_lengths_cum, max_n)
    update_op = _calc_bleu_score(update_num_op, update_denom_op, update_cand_op, update_ref_op, max_n)

    return bleu, update_op


def _get_ngram_counter(sentence, max_n):
    ngrams = collections.Counter()
    for n in range(1, max_n + 1):
        for word_idx in range(0, len(sentence) - n + 1):
            ngram = tuple(sentence[word_idx: word_idx + n])
            ngrams[ngram] += 1
    return ngrams


def _modified_precision(candidate, reference, max_n):
    cand_counter = _get_ngram_counter(candidate, max_n)
    ref_counter = _get_ngram_counter(reference, max_n)

    clipped_counts = {ngram: min(cand_counter[ngram], ref_counter[ngram]) for ngram in cand_counter}

    clipped_counts_by_length = np.zeros(max_n)
    num_cand_ngrams_by_length = np.zeros(max_n)

    for ngram in clipped_counts:
        clipped_counts_by_length[len(ngram) - 1] += clipped_counts[ngram]
    for ngram in cand_counter:
        num_cand_ngrams_by_length[len(ngram) - 1] += cand_counter[ngram]

    return clipped_counts_by_length, num_cand_ngrams_by_length


def _get_bleu_score_params(candidates, references, max_n):
    cand_lengths = 0
    ref_lengths = 0
    clipped_counts_by_length_cum = np.zeros(max_n, dtype=np.float32)
    num_cand_ngrams_by_length_cum = np.zeros(max_n, dtype=np.float32)

    for candidate, reference in zip(candidates, references):
        # delete trailing zeros from padding
        ref = np.trim_zeros(reference, 'b')
        cand = np.trim_zeros(candidate, 'b')
        cand_lengths += len(cand)
        ref_lengths += len(ref)
        clipped_counts_by_length, num_cand_ngrams_by_length = _modified_precision(cand, ref, max_n)
        clipped_counts_by_length_cum += clipped_counts_by_length
        num_cand_ngrams_by_length_cum += num_cand_ngrams_by_length

    return clipped_counts_by_length_cum, num_cand_ngrams_by_length_cum, np.float32(cand_lengths), np.float32(ref_lengths)


def _calc_bleu_score(numerators, denominators, cand_lengths, ref_lengths, max_n):
    precisions = _get_precisions(numerators, denominators, max_n)
    prec_log_mean = _get_mean_precs(precisions)
    bp = _calc_brevity_penalty(ref_lengths, cand_lengths)
    bleu = bp * prec_log_mean
    return bleu


def _get_mean_precs(precisions):
    mask = tf.cast(precisions, tf.bool)
    mask.set_shape([None])
    non_zero_p = tf.boolean_mask(precisions, mask)
    if tf.shape(non_zero_p)[0] == 0:
        return tf.constant(0, dtype=tf.float32)
    log_prec = tf.map_fn(lambda x: tf.log(x), non_zero_p)
    sum_prec = tf.reduce_sum(log_prec)
    max_n = tf.cast(tf.shape(precisions)[0], tf.float32)
    mean = tf.cond(tf.equal(tf.shape(non_zero_p)[0], 0), lambda: tf.constant(0, dtype=tf.float32),
                   lambda: tf.exp(sum_prec / max_n))
    return mean


def _get_precisions(numerators, denominators, max_n):
    precisions = []

    invcnt = tf.constant(1, dtype=tf.float32)

    def denom(sm):
        return sm, tf.constant(0, dtype=tf.float32)

    def num(sm, _denom):
        return sm * tf.constant(2, dtype=tf.float32), tf.constant(1, dtype=tf.float32) / (sm * tf.constant(2, dtype=tf.float32) * _denom)

    def default(sm, _num, _denom):
        return sm, _num / _denom

    for i in range(max_n):
        invcnt, precision = tf.case(OrderedDict([
            (tf.equal(denominators[i], tf.constant(0, tf.float32)), lambda: denom(invcnt)),
            (tf.equal(numerators[i], tf.constant(0, tf.float32)), lambda: num(invcnt, denominators[i])),
        ]), default=lambda: default(invcnt, numerators[i], denominators[i]))
        precisions.append(precision)

    return tf.stack(precisions)


def smoothing(invcnt, denominator):
    invcnt *= 2
    prec = 1.0 / (invcnt * denominator)
    return invcnt, prec


def _calc_brevity_penalty(ref_len, can_len):
    def greater():
        return tf.constant(1, dtype=tf.float32)

    def equal():
        return tf.constant(0, dtype=tf.float32)

    def doelse(x, y):
        return tf.exp(tf.constant(1, dtype=tf.float32) - (x/y))

    out = tf.case({tf.greater(can_len, ref_len): greater,
                   tf.equal(can_len, tf.constant(0, tf.float32)): equal},
                  default=lambda: doelse(ref_len, can_len), exclusive=True)
    return out