import tensorflow as tf
import tensorflow.contrib.slim as slim

import models.vid2sentence_utils
import models.metrics
import models.mobilenet_v1
import models.mobilenet_v2
import models.inception_v4
import models.optimizer
import models.quantization
import constants.constants as const
import constants.hyper_params as hp

from contextlib import ExitStack


def vid2sentence(features, labels, mode, params):

    # get parameters
    drpt_rate = params['dropout_rate']
    heads = params['num_heads']
    ff_hidden_units = params['feed_forward_hidden_units']
    norm_fn = params['normalization_fn']
    activation = params['activation']
    video_emb_type = params['video_emb_type']
    layer_stack = params['layer_stack']
    num_vocab = params['num_vocab']
    input_data_type = params['input_data_type']
    enc_type = params['enc_type']
    tpu = params['tpu']
    bfloat16_mode = params['bfloat16_mode']

    with ExitStack() as context_stack:
        # set bfloat context
        if bfloat16_mode:
            context_stack.enter_context(tf.contrib.tpu.bfloat16_scope())

        assert enc_type in const.EncoderType, "Unknown encoder type: {}".format(enc_type)
        assert input_data_type in const.InputDataTypes, "Unknown input data type: {}".format(input_data_type)

        video_ids = features['video_id']
        features = features['video']

        if params.get('with_labels', True):
            # delete unused last dimension from label_ids
            label_ids, labels = tf.squeeze(labels['label_ids'], [-1]), labels['labels']
            label_emb_size = labels.get_shape()[-1].value
            # shift ids left -- deleting start token as opposed to shift features right
            # left_shift_ids = tf.slice(label_ids, tf.constant([0, 1]), tf.constant([-1, -1]))
            # label_ids = tf.pad(left_shift_ids, tf.constant([[0, 0], [0, 1]]), constant_values=0)

        else:
            label_ids, labels = None, None
            label_emb_size = params['pred_label_emb_size']

        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False

        # encoder
        enc, enc_mask = _encoder(features=features,
                                 label_emb_size=label_emb_size,
                                 is_training=is_training,
                                 drpt_rate=drpt_rate,
                                 layer_stack=layer_stack,
                                 norm_fn=norm_fn,
                                 activation=activation,
                                 video_emb_type=video_emb_type,
                                 input_data_type=input_data_type,
                                 enc_type=enc_type,
                                 num_heads=heads,
                                 ff_hidden_units=ff_hidden_units)

        # decoder
        decode = tf.make_template('decode',
                                  _decoder,
                                  enc=enc,
                                  enc_mask=enc_mask,
                                  drpt_rate=drpt_rate,
                                  layer_stack=layer_stack,
                                  heads=heads,
                                  activation=activation,
                                  ff_hidden_units=ff_hidden_units,
                                  is_training=is_training,
                                  norm_fn=norm_fn,
                                  num_vocab=num_vocab)

        if mode == tf.estimator.ModeKeys.PREDICT:

            batch_size = params['pred_batch_size']
            max_sentence = params['pred_max_sentence']
            emb_lookup_table = params['emb_lookup_table']
            word2id = params['word2id']
            beam_size = params['beam_size']
            alpha = params['beam_alpha']

            predictions = _predictor(batch_size, max_sentence, beam_size, decode, mode, emb_lookup_table, tpu, word2id,
                                     num_vocab, alpha, video_ids)

            return predictions

        logits = decode(labels)

    if bfloat16_mode:
        # cast parameters back to 32 bit for training op
        logits = tf.cast(logits, tf.float32)

    with tf.variable_scope('loss'):
        # final linear transformation + softmax
        preds = tf.argmax(logits, axis=-1)
        mask = tf.cast(tf.cast(tf.reduce_sum(labels, axis=-1), tf.bool), dtype=tf.float32)

        if mode == tf.estimator.ModeKeys.TRAIN:
            label_smoothing = params['label_smoothing']
        else:
            label_smoothing = 0
        try:
            weight_mode = params['weight_mode']
        except KeyError:
            weight_mode = None

        one_hot_labels = tf.one_hot(label_ids, depth=num_vocab)
        weights = models.vid2sentence_utils.get_loss_weights(logits, mask, weight_mode)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits, weights, label_smoothing)
        mean_loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask))

    # get metrics
    if tpu:
        # TPUEstimator does not support all operations needed for BLEU
        def metrics_fn(_label_ids, _preds, _mask):
            return {'accuracy': models.metrics.get_accuracy(_label_ids, _preds, _mask), }

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode, loss=mean_loss, eval_metrics=(metrics_fn, [label_ids, preds, mask]))

    else:
        # max_n_gram = params['max_n_gram']
        # TODO: See how to integrate BLEU with TPU - impossible with current tf version

        with tf.variable_scope('metrics'):
            with tf.variable_scope('acc'):
                acc, acc_op = models.metrics.get_accuracy(label_ids, preds, mask)
            # with tf.variable_scope('bleu'):
            #     bleu, bleu_op = models.metrics.bleu_score(preds, label_ids, max_n_gram)
        # tf.summary.scalar('bleu_op', bleu_op)
        tf.summary.scalar("acc_op", acc_op)
        tf.summary.scalar("mean_loss", mean_loss)

        metrics = {'acc_eval': (acc, acc_op), }
                   # 'bleu_score_eval': (bleu, bleu_op)}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=mean_loss, eval_metric_ops=metrics)

    # create training op
    assert mode == tf.estimator.ModeKeys.TRAIN

    # freeze pre-trained weights
    train_vars = tf.trainable_variables()
    if video_emb_type == const.VideoEmbeddingType.InceptionV4:
        exc_vars = [var for var in train_vars if 'InceptionV4' not in var.name]
    elif video_emb_type == const.VideoEmbeddingType.MobilenetV1:
        exc_vars = [var for var in train_vars if 'MobilenetV1' not in var.name]
    else:
        exc_vars = train_vars

    with tf.variable_scope('train_op'):
        optimizer_type = params['optimizer_type']
        learning_rate = params['learning_rate']

        if optimizer_type == const.OptimizerType.Adam:
            beta1 = params['adam_beta1']
            beta2 = params['adam_beta2']
            epsilon = params['adam_epsilon']

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)

        elif optimizer_type == const.OptimizerType.Adafactor:
            decay_rate = params['adafactor_decay_rate']
            beta1 = params['adafactor_beta1']
            epsilon1 = params['adafactor_epsilon1']
            epsilon2 = params['adafactor_epsilon2']
            if bfloat16_mode:
                parameter_encoding = models.quantization.EighthPowerEncoding()
            else:
                parameter_encoding = None

            optimizer = models.optimizer.AdafactorOptimizer(multiply_by_parameter_scale=True,
                                                            learning_rate=learning_rate,
                                                            decay_rate=decay_rate,
                                                            beta1=beta1,
                                                            epsilon1=epsilon1,
                                                            epsilon2=epsilon2,
                                                            parameter_encoding=parameter_encoding)
        else:
            raise ValueError("Unknown optimizer type {}".format(optimizer_type))

        if tpu:
            use_tpu = params['use_tpu']

            if use_tpu:
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

            return tf.contrib.tpu.TPUEstimatorSpec(mode,
                                                   loss=mean_loss,
                                                   train_op=optimizer.minimize(mean_loss,
                                                   global_step=tf.train.get_global_step(),
                                                   var_list=exc_vars))

        else:
            train_op = optimizer.minimize(mean_loss, global_step=tf.train.get_global_step(), var_list=exc_vars)

            return tf.estimator.EstimatorSpec(mode, loss=mean_loss, train_op=train_op)


def _conv_encoder_stack(inputs,
                        layer_stack,
                        is_training,
                        dropout_rate,
                        norm_fn,
                        kernel_height_list=None,
                        output_dim_list=None):
    """
    Applies a 1D convolution and Gated Linear Units as non-linearity in a specified number of layers
    """

    if kernel_height_list is None:
        if inputs.get_shape().ndims == 5:
            kernel_height_list = layer_stack * [(3, 1, 1)]
        elif inputs.get_shape().ndims == 3:
            kernel_height_list = layer_stack * [3]
    if output_dim_list is None:
        output_dim_list = layer_stack * [inputs.get_shape()[-1].value]

    enc = inputs
    for layer_id in range(layer_stack):
        with tf.variable_scope('layer_{}'.format(layer_id)):
            output_dim = output_dim_list[layer_id]
            residuals = enc

            enc = tf.layers.dropout(enc, dropout_rate, training=is_training)

            # output dimension needs to be doubled for the gated linear units to work
            if inputs.get_shape().ndims == 5:
                enc = tf.layers.conv3d(enc, 2 * output_dim, kernel_height_list[layer_id], 1, padding='same',
                                       activation=None)
            elif inputs.get_shape().ndims == 3:
                enc = tf.layers.conv1d(enc, 2 * output_dim, kernel_height_list[layer_id], 1, padding='same',
                                       activation=None)
            # activation
            enc = models.vid2sentence_utils.gated_linear_units(enc)

            enc += residuals
            enc = norm_fn(enc)

    return enc


def _mult_encoder_stack(enc,
                        enc_mask,
                        layer_stack,
                        num_heads,
                        ff_hidden_units,
                        is_training,
                        dropout_rate,
                        norm_fn,
                        activation,):

    # multihead attention needs 3 dimensions
    if enc.get_shape().ndims == 5:
        enc = models.vid2sentence_utils.reshape_5to3_dimensions(enc)
        adapt_enc_mask = models.vid2sentence_utils.reshape_4to3_dimensions(enc_mask)
    else:
        adapt_enc_mask = enc_mask

    for layer_id in range(layer_stack):
        with tf.variable_scope('layer_{}'.format(layer_id)):
            enc_attn, enc = models.vid2sentence_utils.multihead_attention(queries=enc,
                                                                          keys=enc,
                                                                          query_mask=adapt_enc_mask,
                                                                          key_mask=adapt_enc_mask,
                                                                          dropout_rate=dropout_rate,
                                                                          heads=num_heads,
                                                                          normalize_fn=norm_fn,
                                                                          temp_masking=False,
                                                                          is_training=is_training,
                                                                          scope='multihead_self_attention')

            enc = models.vid2sentence_utils.feed_forward(enc, norm_fn, activation, hidden_units=ff_hidden_units)

    return enc


def _mult_decoder_stack(dec,
                        enc,
                        dec_mask,
                        enc_mask,
                        layer_stack,
                        num_heads,
                        ff_hidden_units,
                        is_training,
                        dropout_rate,
                        norm_fn,
                        activation):

    # multihead attention needs 3 dimensions
    if enc.get_shape().ndims == 5:
        enc = models.vid2sentence_utils.reshape_5to3_dimensions(enc)
        adapt_enc_mask = models.vid2sentence_utils.reshape_4to3_dimensions(enc_mask)
    else:
        adapt_enc_mask = enc_mask

    for layer_id in range(layer_stack):
        with tf.variable_scope('layer_{}'.format(layer_id)):
            dec_attn, dec = models.vid2sentence_utils.multihead_attention(queries=dec,
                                                                          keys=dec,
                                                                          query_mask=dec_mask,
                                                                          key_mask=dec_mask,
                                                                          dropout_rate=dropout_rate,
                                                                          heads=num_heads,
                                                                          normalize_fn=norm_fn,
                                                                          temp_masking=True,
                                                                          is_training=is_training,
                                                                          scope='multihead_self_attention')

            com_attn, dec = models.vid2sentence_utils.multihead_attention(queries=dec,
                                                                          keys=enc,
                                                                          query_mask=dec_mask,
                                                                          key_mask=adapt_enc_mask,
                                                                          dropout_rate=dropout_rate,
                                                                          heads=num_heads,
                                                                          normalize_fn=norm_fn,
                                                                          temp_masking=False,
                                                                          is_training=is_training,
                                                                          scope='multihead_main_attention')

            dec = models.vid2sentence_utils.feed_forward(dec, norm_fn, activation, hidden_units=ff_hidden_units)

    return dec


def _encoder(features, label_emb_size, is_training, drpt_rate, layer_stack, norm_fn, activation, video_emb_type,
             input_data_type, enc_type, num_heads=None, ff_hidden_units=None):

    enc = None
    enc_mask = None
    if input_data_type == const.InputDataTypes.PreprocessedVideoJpg:
        enc = features
        enc_mask = models.vid2sentence_utils.get_frame_pad_mask(features)
    else:
        if video_emb_type == const.VideoEmbeddingType.SpatialCnn:
            spatialnet = _spatial_cnn
        elif video_emb_type == const.VideoEmbeddingType.InceptionV4:
            # get convolution layer from inception v4
            spatialnet = _inception_v4
        elif video_emb_type == const.VideoEmbeddingType.MobilenetV1:
            # get convolution layer from mobilenet v1
            spatialnet = lambda feat, scope, layer='Conv2d_13_pointwise': _mobilenet_v1(feat, is_training, scope, layer)
        elif video_emb_type == const.VideoEmbeddingType.MobilenetV2:
            # get convolution layer from mobilenet v2
            spatialnet = lambda feat, scope, layer='Conv2d_13_pointwise': _mobilenet_v2(feat, is_training, scope, layer)
        if input_data_type == const.InputDataTypes.VideoJpg:
            enc = spatialnet(features, 'spatial_net')
            enc_mask = models.vid2sentence_utils.get_frame_pad_mask(features)
        if input_data_type == const.InputDataTypes.OpticalFlowVideoJpg:
            jpg_enc = spatialnet(features['video'], 'spatial_net', 'Conv2d_12_depthwise')
            flow_stacks = features['flow_stacks']
            flow_enc = _mobilenet_v1(flow_stacks, is_training, 'temporal_net', 'Conv2d_5_depthwise')
            # enc = jpg_enc
            enc = tf.concat([jpg_enc, flow_enc], -1)
            enc_mask = models.vid2sentence_utils.get_frame_pad_mask(features['video'])

    with tf.variable_scope('encoder'):
        # with tf.variable_scope('reduce_dimensions'):
        #     # reduce model output filters to sentence embedding size
        #     enc = models.vid2sentence_utils.reduce_channels_to_size(enc, label_emb_size, activation)

        enc += models.vid2sentence_utils.sinus_pos_encoding(enc)

        enc = tf.layers.dropout(enc, drpt_rate, training=is_training)
        enc_mask = models.vid2sentence_utils.adapt_frame_mask_to_encoding(enc_mask, enc.get_shape().as_list())

        if enc_type == const.EncoderType.Convolution:
            enc = _conv_encoder_stack(inputs=enc,
                                      layer_stack=layer_stack,
                                      is_training=is_training,
                                      dropout_rate=drpt_rate,
                                      norm_fn=norm_fn)  # TODO add query key encoding if necessary and adapt enc mask

        elif enc_type == const.EncoderType.MultiHeadAttention:
            _mult_encoder_stack(enc=enc,
                                enc_mask=enc_mask,
                                layer_stack=layer_stack,
                                num_heads=num_heads,
                                ff_hidden_units=ff_hidden_units,
                                is_training=is_training,
                                dropout_rate=drpt_rate,
                                norm_fn=norm_fn,
                                activation=activation)

        return enc, enc_mask


def _decoder(labels, enc, enc_mask, drpt_rate, layer_stack, heads, activation, ff_hidden_units, is_training, norm_fn,
             num_vocab):

    dec = labels
    # dec = tf.print(labels, [labels], 'labels', summarize=3200)
    dec_mask = models.vid2sentence_utils.get_label_pad_mask(labels)

    # dec = tf.print(dec, [dec], 'dec1', summarize=3200)
    dec += models.vid2sentence_utils.sinus_pos_encoding(dec)
    # dec = tf.print(dec, [dec], 'dec1', summarize=3200)

    dec = tf.layers.dropout(dec, drpt_rate, training=is_training)

    dec = _mult_decoder_stack(dec=dec,
                              enc=enc,
                              dec_mask=dec_mask,
                              enc_mask=enc_mask,
                              layer_stack=layer_stack,
                              num_heads=heads,
                              ff_hidden_units=ff_hidden_units,
                              is_training=is_training,
                              dropout_rate=drpt_rate,
                              norm_fn=norm_fn,
                              activation=activation)

    logits = tf.layers.dense(dec, num_vocab, activation=activation)

    return logits


def _predictor(batch_size, max_sentence, beam_size, decode_fn, mode, emb_lookup_table, tpu, word2id, vocab_size, alpha, video_ids):

    # labels, logits = _greedy_decoder(batch_size, max_words, decode_fn, emb_lookup_table, word2id)
    labels, log_probs, log_probs_softmax = beam_search_decoder(beam_size, batch_size, decode_fn, max_sentence,
                                                               emb_lookup_table, word2id, vocab_size, alpha)
    probs = tf.exp(log_probs)

    predictions = {
        'labels': labels,
        'probs': probs,
        'probs_softmax': log_probs_softmax,
        'video_ids': video_ids
    }
    if tpu:
        return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)
    else:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def _greedy_decoder(batch_size, max_words, decode_fn, emb_lookup_table, word2id):
    start_id = word2id[u"<s>"]
    label_ids = tf.zeros((batch_size, max_words), tf.int64)
    label_size = label_ids.get_shape().as_list()[1]
    logits = None

    for i in range(label_size):
        shift_right_ids = tf.slice(label_ids, [0, 0], [-1, label_size - 1])
        shift_right_ids = tf.pad(shift_right_ids, tf.constant([[0, 0], [1, 0]]), constant_values=start_id)
        labels = tf.map_fn(lambda x: tf.nn.embedding_lookup(tf.convert_to_tensor(emb_lookup_table, tf.float32), x),
                           shift_right_ids, dtype=tf.float32)
        logits = decode_fn(labels)
        label_ids = tf.argmax(logits, axis=-1)

    return label_ids, logits


def beam_search_decoder(beam_size, batch_size, decode_fn, max_sentence, emb_lookup_table, word2id, vocab_size, alpha):
    # i = current word length
    start_id = word2id[u"<s>"]
    eos_id = word2id[u"</s>"]
    unk_id = word2id[u"unk"]

    alive_seqs = None
    alive_log_probs = tf.zeros((batch_size, 1), dtype=tf.float32)
    # 2 X 3

    for i in range(max_sentence):
        beam_candidates_list = []

        for j in range(beam_size):
            if i == 0:
                ids = tf.zeros((batch_size, max_sentence), dtype=tf.int64)
            else:
                ids = tf.squeeze(
                    tf.slice(tf.transpose(alive_seqs, [1, 0, 2]), [j, 0, 0], tf.constant([1, -1, -1])), [0])
            labels_size = ids.get_shape().as_list()[1]
            pad_ids = tf.pad(ids, [[0, 0], [0, max_sentence - labels_size]], constant_values=unk_id)
            sliced_ids = tf.slice(pad_ids, tf.constant([0, 0]), [-1, max_sentence - 1])
            shift_right_ids = tf.pad(sliced_ids, tf.constant([[0, 0], [1, 0]]), constant_values=start_id)
            # 2 X 4

            # get embedded label ids
            labels = tf.map_fn(lambda x: tf.nn.embedding_lookup(tf.convert_to_tensor(emb_lookup_table, tf.float32), x),
                               shift_right_ids, dtype=tf.float32)
            # 2 X 4 X 2

            # get logits from model
            logits = decode_fn(labels)
            # 2 X 4 X 6

            # get relevant logits
            rel_logits = tf.slice(logits, tf.constant([0, i, 0]), tf.constant([-1, 1, -1]))
            current_logits = tf.squeeze(rel_logits, [1])
            # 2 X 6

            beam_candidates_list.append(current_logits)

            # start with only 1 beam
            if i == 0:
                break

        # 3 X 2 X 6
        beam_candidates = tf.transpose(tf.stack(beam_candidates_list), perm=[1, 0, 2])
        # 2 X 3 X 6

        # normalize log probabilities
        log_probs = beam_candidates - tf.reduce_logsumexp(beam_candidates, axis=-1, keepdims=True)
        # 2 X 3 X 6

        # multiply with beam probs or add log(probs) and apply length penalty
        shaped_beam_log_probs = tf.tile(tf.expand_dims(alive_log_probs, -1), (1, 1, vocab_size))
        log_probs = log_probs + shaped_beam_log_probs
        length_penalty = tf.pow(((5. + i) / 6.), alpha)
        scores = log_probs / length_penalty
        # 2 X 3 X 6

        # flatten scores
        flat_scores = tf.reshape(scores, [batch_size, -1])
        # 2 X 18

        topk_scores, topk_ids = tf.nn.top_k(flat_scores, k=beam_size * 2)

        topk_log_probs = topk_scores * length_penalty
        beam_pos = topk_ids // vocab_size
        topk_ids %= vocab_size

        batch_pos = tf.range(batch_size * beam_size * 2) // (beam_size * 2)
        batch_pos = tf.reshape(batch_pos, [batch_size, (beam_size * 2)])

        topk_indices = tf.stack([batch_pos, beam_pos], axis=2)

        if i == 0:
            topk_seq = tf.expand_dims(topk_ids, axis=-1)
        else:
            topk_seq = tf.gather_nd(alive_seqs, topk_indices)
            topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2)

        finished = tf.equal(topk_ids, eos_id)

        alive_scores = topk_scores + tf.to_float(finished) * tf.float32.min
        _, alive_topk_ids = tf.nn.top_k(alive_scores, k=beam_size)
        alive_batch_pos = tf.range(batch_size * beam_size) // beam_size
        alive_batch_pos = tf.reshape(alive_batch_pos, [batch_size, beam_size])
        alive_topk_indices = tf.stack([alive_batch_pos, alive_topk_ids], axis=2)

        alive_seqs = tf.gather_nd(topk_seq, alive_topk_indices)
        alive_log_probs = tf.gather_nd(topk_log_probs, alive_topk_indices)

        finished_scores = topk_scores + (1. - tf.to_float(finished)) * tf.float32.min
        if i == 0:
            fin_candidate_scores = finished_scores
            fin_candidate_seqs = topk_seq
            fin_candidate_log_probs = topk_log_probs
        else:
            fin_candidate_scores = tf.concat([topk_finished_score, finished_scores], axis=-1)
            fin_candidate_seqs = tf.concat([tf.pad(topk_finished_seqs, [[0, 0], [0, 0], [0, 1]],
                                                   constant_values=unk_id), topk_seq], axis=1)
            fin_candidate_log_probs = tf.concat([topk_finished_log_probs, topk_log_probs], axis=-1)

        _, finished_topk_ids = tf.nn.top_k(fin_candidate_scores, k=beam_size)
        finished_batch_pos = tf.range(batch_size * beam_size) // (beam_size)
        finished_batch_pos = tf.reshape(finished_batch_pos, [batch_size, (beam_size)])
        finished_topk_indeces = tf.stack([finished_batch_pos, finished_topk_ids], axis=2)

        topk_finished_seqs = tf.gather_nd(fin_candidate_seqs, finished_topk_indeces)
        topk_finished_score = tf.gather_nd(fin_candidate_scores, finished_topk_indeces)
        topk_finished_log_probs = tf.gather_nd(fin_candidate_log_probs, finished_topk_indeces)

    topk_finished_softmax = tf.nn.softmax(topk_finished_log_probs)
    return topk_finished_seqs, topk_finished_log_probs, topk_finished_softmax


def _inception_v4(inputs, scope=None, layer='Mixed_6h'):
    with slim.arg_scope(models.inception_v4.inception_v4_arg_scope()):
        # num_classes=None so endpoints are returned without softmax computation
        return tf.map_fn(lambda x: models.inception_v4.inception_v4(
            x, num_classes=None, is_training=False,
            reuse=tf.AUTO_REUSE, create_aux_logits=False)[1][layer], inputs)


def _mobilenet_v1(inputs, is_training, scope, layer):
    # with tf.variable_scope(scope):
    with slim.arg_scope(models.mobilenet_v1.mobilenet_v1_arg_scope(is_training=is_training)):
            # with tf.variable_scope('mobile_net_v1'):
            output = tf.map_fn(lambda x: models.mobilenet_v1.mobilenet_v1(x, num_classes=1001, depth_multiplier=0.25,
                                                               reuse=tf.AUTO_REUSE)[1][layer], inputs)
    return output


def _mobilenet_v2(inputs, is_training, scope, layer='layer_18/output'):
    # with tf.variable_scope(scope):
    with slim.arg_scope(models.mobilenet_v2.training_scope(is_training=is_training)):
            # with tf.variable_scope('mobile_net_v2'):
            output = tf.map_fn(lambda x: models.mobilenet_v2.mobilenet(x, num_classes=1001,
                                                               reuse=tf.AUTO_REUSE)[1][layer], inputs)
    return output


def _spatial_cnn(inputs):
    with tf.variable_scope('spatial_cnn'):
        model = models.vid2sentence_utils.vgg16_3d('spatial')
        output = inputs
        for layer in model:
            output = layer(output)

        return output


def _temporal_cnn(flow_x, flow_y):
    with tf.variable_scope('temporal_cnn'):
        model = models.vid2sentence_utils.vgg16_3d('temporal', hp.FLOW_RESIZE_FACTOR)

        inputs = tf.concat([flow_x, flow_y], axis=-1)
        output = inputs
        for layer in model:
            output = layer(output)

        return output
