import tensorflow as tf
import numpy as np
import constants.constants as const


def reduce_channels_to_size(inputs, size, activation):
    """
    Applies a One by One Convolution to alter the channel size.
    """
    conv = tf.layers.Conv3D(size, (1, 1, 1), (1, 1, 1), padding='valid', name='reduce_conv', activation=activation)
    return conv(inputs)


def reshape_5to3_dimensions(tensor):
    """
    Reshapes tensor so that all surplus dimensions get shaped into the middle dimension,
    even if the first dimension is dynamic
    i.e.: [?, 120, 4, 4, 64] -> [[?, 1920, 64]
    """
    shape = tensor.get_shape().as_list()
    assert len(shape) == 5, "reshape_5to3_dimensions accepts only tensors with 5 dimensions"
    mul_dim = 1
    for dim in shape[1:-1]:
        mul_dim *= dim
    out_shape = [-1, mul_dim, shape[-1]]
    tensor = tf.reshape(tensor, out_shape)

    return tensor


def reshape_4to3_dimensions(tensor):
    """
    Reshapes tensor so that all surplus dimensions get shaped into the middle dimension,
    even if the first dimension is dynamic
    i.e.: [?, 120, 4, 4, 64] -> [[?, 1920, 64]
    """
    shape = tensor.get_shape().as_list()
    assert len(shape) == 4, "reshape_4to3_dimensions accepts only tensors with 4 dimensions"
    mul_dim = 1
    for dim in shape[1:]:
        mul_dim *= dim
    out_shape = [-1, mul_dim]
    tensor = tf.reshape(tensor, out_shape)

    return tensor


def sinus_pos_encoding(inputs):
    """
    Computes a sinusoidal grid of the input's size.
    """
    shape = inputs.get_shape().as_list()
    if len(shape) == 3:
        _, length, d_model = inputs.get_shape().as_list()
    else:
        _, length, height, weight, d_model = inputs.get_shape().as_list()
        length = length * height * weight

    pos_enc = np.array([[pos / pow(10000, 2 * i / d_model) for i in range(d_model)]
                        if pos != 0 else np.zeros(d_model) for pos in range(length)])

    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])

    pos_enc = tf.convert_to_tensor(pos_enc, dtype=inputs.dtype)
    pos_enc = tf.reshape(pos_enc, shape[1:])
    out_mult = [1] * len(shape)
    out_mult[0] = tf.shape(inputs)[0]
    pos_enc = tf.tile(tf.expand_dims(pos_enc, 0), out_mult)

    return pos_enc


def gated_linear_units(inputs):
    """
    Non linearity as described in: https://arxiv.org/abs/1612.08083
    """
    input_shape = inputs.get_shape().as_list()
    input_pass = inputs[..., 0:input_shape[-1] // 2]
    input_gate = inputs[..., input_shape[-1] // 2:]
    input_gate = tf.sigmoid(input_gate)

    return tf.multiply(input_pass, input_gate)


def get_temporal_mask(inputs):
    """
    Returns a lower diagonal matrix effectively masking one less word for every row.
    For example:  [[[1. 2. 3.]      ->   [[[1. 0. 0.]
                    [4. 5. 6.]             [1. 1. 0.]
                    [7. 8. 9.]]            [1. 1. 1.]]
                                    ->
                   [[1. 2. 3.]            [[1. 0. 0.]
                    [4. 5. 6.]             [1. 1. 0.]
                    [7. 8. 9.]]]    ->     [1. 1. 1.]]
    """
    lower = tf.matrix_band_part(tf.ones_like(inputs), -1, 0)
    # diag = tf.matrix_band_part(tf.ones_like(inputs), 0, 0)
    # output = lower - diag
    output = lower
    return output


def layer_normalization(inputs, scope='layer_normalization', reuse=None, epsilon=1e-5):
    """
    Layer normalization as described in https://arxiv.org/pdf/1607.06450.pdf. Reduces importance of proper
    initialization.
    """
    with tf.variable_scope(scope, reuse=reuse):
        mean, var = tf.nn.moments(inputs, [-1], keep_dims=True)
        shape = inputs.get_shape()[-1]
        gain = tf.get_variable('gain', shape, initializer=tf.ones_initializer(), dtype=inputs.dtype)
        bias = tf.get_variable('bias', shape, initializer=tf.zeros_initializer(), dtype=inputs.dtype)

        ln = gain * (inputs - mean) / tf.sqrt(var + epsilon) + bias

    return ln


def feed_forward(inputs, norm_fn, activation, hidden_units=None, scope='feed-forward', reuse=None):
    """
    Position wise fully connected feed forward network consisting of two linear transformations with a RELU activation
    function in between.
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        d_model = inputs.get_shape()[-1].value
        if hidden_units is None:
            hidden_units = 4 * d_model
        inner_layer = tf.layers.conv1d(inputs, filters=hidden_units, kernel_size=1, activation=activation)
        outputs = tf.layers.conv1d(inner_layer, filters=d_model, kernel_size=1, activation=None)

        # add & norm
        outputs += inputs
        outputs = norm_fn(outputs)

    return outputs


def multihead_attention(queries,
                        keys,
                        query_mask,
                        key_mask,
                        dropout_rate,
                        heads,
                        temp_masking,
                        normalize_fn,
                        is_training,
                        scope):
    """

    """
    with tf.variable_scope(scope):

        # get embedding dimensions
        d_model = queries.get_shape()[-1].value

        # linear projections - equivalent to having number of heads times smaller linear projections
        # reduce embedding dimensionality by dividing by number of heads
        q_lin_trans = tf.layers.dense(queries, d_model, use_bias=False)  # mb_size X len_q X d_model
        k_lin_trans = tf.layers.dense(keys, d_model, use_bias=False)     # ” X ” X ”
        v_lin_trans = tf.layers.dense(keys, d_model, use_bias=False)     # ” X ” X ”

        # slice channels into number of heads portions and concat them on the first dim
        q_hslice = tf.concat(tf.split(q_lin_trans, heads, axis=-1), axis=0)   # mb_size*heads X len_q X d_model/heads
        k_hslice = tf.concat(tf.split(k_lin_trans, heads, axis=-1), axis=0)   # ” X ” X ”
        v_hslice = tf.concat(tf.split(v_lin_trans, heads, axis=-1), axis=0)   # ” X ” X ”

        # get masks for zero padded trails
        query_mask, key_mask = get_multattn_sub_masks(query_mask, key_mask, heads)

        # apply attention mechanism
        attn, outputs = scaled_dot_product_attention(queries=q_hslice,
                                                     keys=k_hslice,
                                                     values=v_hslice,
                                                     is_training=is_training,
                                                     query_mask=query_mask,
                                                     key_mask=key_mask,
                                                     temp_mask=temp_masking,
                                                     dropout_rate=dropout_rate)  # mb_size*heads X len_q X d_model/heads

        # slice first dim and concat channels - revert slicing into heads
        outputs = tf.concat(tf.split(outputs, heads, axis=0), axis=-1)   # mb_size X len_q X d_model

        # add & norm (residual and normalization)
        outputs += queries
        outputs = normalize_fn(outputs)

        return attn, outputs


def get_multattn_sub_masks(query_mask, key_mask, heads):
    """
    Get binary masks to account for zero-padded input due to padded batched datasets.
    """
    query_mask_sub = tf.tile(query_mask, [heads, 1])    # (h*mb_size X len_q)
    key_mask_sub = tf.tile(key_mask, [heads, 1])         # (h*mb_size X len_k)

    query_mask_sub = tf.tile(tf.expand_dims(query_mask_sub, axis=-1), [1, 1, key_mask.get_shape().as_list()[1]])
    # (h*mb_size X len_q X len_k)
    key_mask_sub = tf.tile(tf.expand_dims(key_mask_sub, axis=1), [1, query_mask.get_shape().as_list()[1], 1])
    # (h*mb_size X len_q X len_k)

    return query_mask_sub, key_mask_sub


def scaled_dot_product_attention(queries,
                                 keys,
                                 values,
                                 is_training,
                                 dropout_rate,
                                 query_mask,
                                 key_mask,
                                 temp_mask,
                                 scope='scaled_dot_product_attention'):
    """
    Implements attention as used in 'Attention is all you need': soft-max((QK^T)/sqrt(d_K))V.
    """
    with tf.variable_scope(scope):

        attn = tf.matmul(queries, keys, transpose_b=True)    # mb_size X len_q X len_k

        # scale
        attn /= tf.sqrt(tf.cast(keys.get_shape().as_list()[-1], attn.dtype))

        # for decoder self attention with teacher forcing, inputs must be masked to the respective point in time
        if temp_mask:
            attn_mask = get_temporal_mask(attn)
            mask = key_mask * attn_mask
        else:
            mask = key_mask
        padding = tf.ones_like(attn) * attn.dtype.min
        attn = tf.where(tf.equal(mask, 0), padding, attn)

        # get distribution
        attn = tf.nn.softmax(attn)  # mb_size X len_q X len_k

        # query masking
        attn *= query_mask

        # regularization
        reg_attn = tf.layers.dropout(attn, dropout_rate, training=is_training)

        # apply attention
        output = tf.matmul(reg_attn, values)  # mb_size X len_q X d_model

        return attn, output


def label_smoothing(labels, epsilon=0.1):
    """
    Label smoothing as seen in https://arxiv.org/pdf/1512.00567.pdf . Increases generalization through introduction
    of uncertainty.
    """
    k = labels.get_shape().as_list()[-1]
    return ((1 - epsilon) * labels) + (epsilon / k)


def get_frame_pad_mask(input_to_mask):
    return tf.sign(tf.abs(tf.reduce_sum(input_to_mask, axis=[-3, -2, -1])))


def adapt_frame_mask_to_encoding(frame_mask, encoding_shape):
    height, width = encoding_shape[-3], encoding_shape[-2]
    adapted_mask = tf.tile(tf.expand_dims(tf.expand_dims(frame_mask, -1), -1), [1, 1, height, width])
    return adapted_mask


def get_label_pad_mask(input_to_mask):
    return tf.sign(tf.abs(tf.reduce_sum(input_to_mask, axis=-1)))


def vgg16_3d(name, reduce_factor=-1):
    """
    Simple CNN based on VGG16 but for video data. Skipping first layers, if corresponding reduce factor is given.
    """
    threshold = reduce_factor
    if reduce_factor != -1:
        threshold = np.log(reduce_factor) / np.log(2)
        assert threshold == int(threshold), "reduce factor needs to be in 2^x"

    model_seq = []
    out_filters = [64, 128, 256, 512, 512]
    for i, out_nodes in enumerate(out_filters):
        if i >= threshold:
            model_seq.append(tf.layers.Conv3D(out_nodes, (1, 3, 3), 1, padding='same', activation=tf.nn.selu,
                                              name='{}_conv{}'.format(name, i)))
            model_seq.append(tf.layers.MaxPooling3D((1, 2, 2), (1, 2, 2), name='{}_max_pool{}'.format(name, i)))
    return model_seq


def get_loss_weights(logits, mask=None, weight_mode=None):
    """
    Returns negative linear weights, for example:
    input dims 2,3,X -> [[3, 2, 1], [3, 2, 1]]
    :param mask:
    :param logits:
    :return:
    """
    static = logits.get_shape().as_list()
    shape = tf.shape(logits)
    batch_size = shape[0]
    word_count = static[1]
    data_type = logits.dtype
    if weight_mode == const.WeightMode.Gauss:
        single_weights = get_gauss_weights(word_count, data_type)
    elif weight_mode == const.WeightMode.Linear:
        single_weights = get_linear_weights(word_count, data_type)
    else:
        single_weights = tf.ones(word_count, dtype=data_type)
    weights = tf.tile(tf.expand_dims(single_weights, axis=0), [batch_size, 1])
    if mask is not None:
        weights *= mask
    return weights


def get_linear_weights(word_count, data_type):
    """
    Returns a tensor of weights, linearly getting smaller and adding up to word_count.
    :param word_count:
    :return:
    """
    linear_list = list(range(word_count, 0, -1))
    norm_dist = [i * word_count / float(sum(linear_list)) for i in linear_list]
    norm_dist = tf.convert_to_tensor(norm_dist, dtype=data_type)
    return norm_dist


def get_gauss_weights(word_count, data_type):
    """
    Returns a tensor of weights, getting smaller according to a half normal distribution (sigma=1) and adding
    up to word_count (approx. bc softmax)
    :param word_count:
    :return:
    """
    dist = tf.contrib.distributions.HalfNormal(scale=float(1))
    dist_list = [dist.prob(float(i)) for i in range(word_count)]
    tensor_dist = tf.convert_to_tensor(dist_list, dtype=data_type)
    softmax_dist = tf.nn.softmax(tensor_dist)
    weights = softmax_dist * word_count
    return weights
