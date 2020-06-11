import tensorflow as tf

def int64_feature(value):
    """
    Wrapper for inserting int64 features into Example proto.
    :param value: single int value
    :return: example proto
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_list_feature(value):
    """
    Wrapper for inserting float features into Example proto.
    :param value: single float value
    :return: example proto
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """
    Wrapper for inserting bytes features into Example proto.
    :param value: single byte value
    :return: example proto
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature_list(values):
    """
    Wrapper for inserting an int64 FeatureList into a SequenceExample proto,
    e.g, sentence in list of ints
    :param values:
    :return:
    """
    return tf.train.FeatureList(feature=[int64_feature(v) for v in values])


def float_list_feature_list(values):
    """
    Wrapper for inserting an float FeatureList into a SequenceExample proto,
    e.g, sentence in embedded list of floats
    :param values:
    :return:
    """
    return tf.train.FeatureList(feature=[float_list_feature(v) for v in values])


def bytes_feature_list(values):
    """
    Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
    e.g, sentence in list of bytes
    :param values:
    :return:
    """
    return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])
