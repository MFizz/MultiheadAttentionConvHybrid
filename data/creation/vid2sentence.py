from tensorflow.python.lib.io import file_io
from pandas.compat import StringIO
import pandas as pd
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import data.creation.word_processing as word_procession
import data.creation.image_processing as image_processing
import data.creation.io_utils as io_utils
import constants.constants as const
import constants.hyper_params as hp
import os
import sys
import logging
import models.inception_v4
import matplotlib
import random
matplotlib.use("agg")
import matplotlib.pyplot as plt
import data.creation.tfr_utils as tfr_utils


class V2SBaseDataset:
    def __init__(self, name,
                 w2v_model,
                 frames_path,
                 opt_frames_path,
                 train_labels_filename,
                 eval_labels_filename,
                 predict_labels_filename,
                 predict_name,
                 model_base_dir,
                 sentence_max,
                 frames_max,
                 max_height,
                 max_width,
                 tfr_ex_num,
                 jpg_skip,
                 opt_flow_bracket):

        model_directory = os.path.join(model_base_dir, name, 'model_emb_{}_negsamp_{}_skipwin_{}'.
                                       format(w2v_model.embedding_size, w2v_model.neg_samples,
                                              w2v_model.skip_word_window))

        self.name = name
        self.model_dir = model_directory
        self.train_labels_filename = train_labels_filename
        self.eval_labels_filename = eval_labels_filename
        self.predict_labels_filename = predict_labels_filename
        self.predict_name = predict_name
        self.frames_path = frames_path
        self.opt_frames_path = opt_frames_path
        self.sentence_max = sentence_max
        self.frames_max = frames_max
        self.max_height = max_height
        self.max_width = max_width
        self.tfr_ex_num = tfr_ex_num
        self.w2v_model = w2v_model
        self.jpg_skip = jpg_skip
        self.opt_flow_bracket = opt_flow_bracket

        logger = logging.getLogger('main.vid2sentence')

        # get train input data
        train_name = "train"
        if self.train_labels_filename:
            self.train_data_path = os.path.join(self.model_dir, train_name)
            self.input_train_data_files = self.get_or_create_dataset(train_name, self.train_data_path,
                                                                     train_labels_filename)
        else:
            logger.info("Dataset will be created without training set, training operations impossible.")

        # get train input data
        eval_name = "eval"
        if self.eval_labels_filename:
            self.eval_data_path = os.path.join(self.model_dir, eval_name)
            self.input_eval_data_files = self.get_or_create_dataset(eval_name, self.eval_data_path,
                                                                    eval_labels_filename)
        else:
            logger.info("Dataset will be created without eval set, evaluation operations impossible.")

        if self.predict_labels_filename:
            self.predict_data_path = os.path.join(self.model_dir, predict_name)
            self.input_predict_data_files = self.get_or_create_dataset(predict_name, self.predict_data_path,
                                                                       predict_labels_filename)
        else:
            logger.info("No prediction set selected.")

        if self.train_labels_filename is None and self.eval_labels_filename is None \
                and self.predict_labels_filename is None:
            logger.info("No file paths given - not able to generate dataset.")

    def create_dataset(self, model_path, tfr_path, label_filename):
        raise NotImplementedError

    def get_or_create_dataset(self, name, model_path, label_filename):
        """
        Looks for input data set for vid2sentence or creates it.
        """
        logger = logging.getLogger('main.vid2sentence')
        # tensorflow bug in makedirs revert when fixed TODO
        tf.gfile.MakeDirs(model_path)
        tfr_files = [os.path.join(model_path, file) for file in tf.gfile.ListDirectory(model_path)
                     if file.endswith(".tfr")]

        if len(tfr_files) == 0:
            logger.info('No Vid2Sentence model {} input files found, now creating at {}'.format(name, model_path))
            tfr_path = os.path.join(model_path, 'senlen_{}_framelen_{}_height_{}_width_{}'
                                    .format(self.sentence_max, self.frames_max, self.max_height, self.max_width))
            tfr_files = self.create_dataset(model_path, tfr_path, label_filename)
        else:
            logger.info('Vid2Sentence model input files used in : {}'.format(model_path))

        return tfr_files

    def generate_embedded_labels_frames(self, labels, optical_flow=False):
        """
        Returns a generator, generating sentence, referring ids and referring video frames + id.
        """

        if labels.shape[1] == 1:
            no_labels = True
        elif labels.shape[1] == 2:
            no_labels = False
        else:
            raise ValueError("Labels need either 1 column for video_id only or 2 for additional label. Where {}"
                             .format(labels.shape[1]))

        for row_idx, row in labels.iterrows():

            if no_labels:
                embedded_sentence, sentence_ids = None, None
            else:
                # get embedded labels
                embedded_sentence, sentence_ids = self.get_embedded_sentence(row['sentence'])

            # get matching frames
            if optical_flow:
                encoded_frames, flows = _get_frames_with_opt_flow(row['video_id'], self.jpg_skip,
                                                                  self.opt_flow_bracket,  self.frames_path,
                                                                  self.opt_frames_path, self.frames_max,
                                                                  self.max_height, self.max_width, True)
                if no_labels:
                    yield row_idx, row['video_id'], encoded_frames, flows
                else:
                    yield row_idx, row['video_id'], embedded_sentence, sentence_ids, encoded_frames, flows
            else:
                encoded_frames = _get_frames(row['video_id'], self.frames_path,
                                             self.frames_max, self.max_height, self.max_width)
                if no_labels:
                    yield row_idx, row['video_id'], encoded_frames
                else:
                    yield row_idx, row['video_id'], embedded_sentence, sentence_ids, encoded_frames

    def get_embedded_sentence(self, sentence):
        """
        Normalizes sentence and embeds it.
        """
        # convert labels into the right format
        norm_sentence = word_procession.norm_sentence(sentence)

        sentence_ids = [self.w2v_model.word2id[word] for word in norm_sentence.split(' ')]

        # get only up to sentence max length
        sentence_ids = sentence_ids[:self.sentence_max]

        embedded_sentence = self.w2v_model.emb_lookup_table[sentence_ids]
        return embedded_sentence, sentence_ids

    def input_data_type(self):
        if isinstance(self, V2SDataset):
            return const.InputDataTypes.VideoJpg
        elif isinstance(self, V2SprecDataset):
            return const.InputDataTypes.PreprocessedVideoJpg
        elif isinstance(self, V2SOpticalFlowDataset):
            return const.InputDataTypes.OpticalFlowVideoJpg
        else:
            return None


class V2SDataset(V2SBaseDataset):
    def __init__(self, name,
                 w2v_model,
                 frames_path,
                 opt_frames_path=None,
                 train_labels_filename=None,
                 eval_labels_filename=None,
                 prediction_labels_filename=None,
                 prediction_name=None,
                 model_base_dir=const.VID2SENTENCE_DEFAULT_MODEL_PATH,
                 sentence_max=hp.V2S_SENTENCE_MAX,
                 frames_max=hp.V2S_FRAMES_MAX,
                 max_height=hp.V2S_MAX_HEIGHT,
                 max_width=hp.V2S_MAX_WIDTH,
                 tfr_ex_num=hp.V2S_TFR_EX_NUM,
                 jpg_skip=hp.JPG_SKIP,
                 opt_flow_bracket=hp.BRACKET
                 ):

        super().__init__(name,
                         w2v_model,
                         frames_path,
                         opt_frames_path,
                         train_labels_filename,
                         eval_labels_filename,
                         prediction_labels_filename,
                         prediction_name,
                         model_base_dir,
                         sentence_max,
                         frames_max,
                         max_height,
                         max_width,
                         tfr_ex_num,
                         jpg_skip,
                         opt_flow_bracket)

    def create_dataset(self, data_path, tfr_path, label_filename):
        """
        Creates input data set for Vid2Sentence which contains already preprocessed word data and jpg frames:
        """
        with tf.gfile.GFile(label_filename, mode='r') as labels_csv:
            tfr_files = []

            if const.GCLOUD:
                labels = read_bucket_csv(label_filename)
            else:
                labels = pd.read_csv(labels_csv, delimiter=';', header=None)

            _, num_columns = labels.shape
            if num_columns == 1:
                cols = ['video_id']
            elif num_columns == 2:
                cols = ['video_id', 'sentence']
            else:
                raise ValueError('Csv file has an invalid amount of colums: {}'.format(num_columns))
            labels.columns = cols

            labels = labels[:256]  # TODO delete

            # logging variables
            sentence_lengths = []
            frame_lengths = []
            frame_widths = []
            frame_heights = []

            # get a generator for labels and features
            data_gen = self.generate_embedded_labels_frames(labels)

            tfr_writer = None

            for row in data_gen:
                if num_columns == 1:
                    row_idx, video_id, encoded_frames = row

                    # encode frames as jpg
                    jpg_frames = [tf.compat.as_bytes(image_processing.encode_jpg(img).tobytes()) for img in
                                  encoded_frames]

                    context = tf.train.Features(feature={'video_id': tfr_utils.int64_feature(video_id)})
                    feature_list = {
                        'out_layers': tfr_utils.bytes_feature_list(jpg_frames)
                    }

                else:
                    row_idx, video_id, embedded_sentence, sentence_ids, encoded_frames = row

                    # encode frames as jpg
                    jpg_frames = [tf.compat.as_bytes(image_processing.encode_jpg(img).tobytes()) for img in
                                  encoded_frames]

                    # get data for logging
                    sentence_lengths.append(len(sentence_ids))
                    num, height, width, channels = encoded_frames.shape
                    frame_lengths.append(num)
                    frame_heights.append(height)
                    frame_widths.append(width)

                    context = tf.train.Features(feature={'video_id': tfr_utils.int64_feature(video_id)})
                    feature_list = {
                        'label': tfr_utils.float_list_feature_list(embedded_sentence),
                        'label_ids': tfr_utils.float_list_feature_list(np.expand_dims(np.asarray(sentence_ids), axis=-1)),
                        'out_layers': tfr_utils.bytes_feature_list(jpg_frames)
                    }

                feature_lists = tf.train.FeatureLists(feature_list=feature_list)

                example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

                # create new tfr file every $tfr_ex_num examples
                if row_idx % self.tfr_ex_num == 0:
                    tfr_file = tfr_path + '_{}.tfr'.format((int(row_idx / self.tfr_ex_num) + 1)
                                                           * self.tfr_ex_num)
                    tfr_files.append(tfr_file)
                    if tfr_writer:
                        tfr_writer.close()
                        sys.stdout.flush()
                    tfr_writer = tf.python_io.TFRecordWriter(tfr_file)
                tfr_writer.write(example.SerializeToString())

            if tfr_writer:
                tfr_writer.close()
                sys.stdout.flush()

            if not const.GCLOUD and num_columns == 2:
                _save_histogram('Sentence Length', sentence_lengths, data_path)
                _save_histogram('Frame Length', frame_lengths, data_path)
                _save_histogram('Frame Width', frame_widths, data_path)
                _save_histogram('Frame Height', frame_heights, data_path)

            return tfr_files


class V2SprecDataset(V2SBaseDataset):
    def __init__(self, name,
                 w2v_model,
                 frames_path,
                 train_labels_filename=None,
                 eval_labels_filename=None,
                 prediction_labels_filename=None,
                 prediction_name=None,
                 model_base_dir=const.VID2SENTENCE_DEFAULT_MODEL_PATH,
                 sentence_max=hp.V2S_SENTENCE_MAX,
                 frames_max=hp.V2S_FRAMES_MAX,
                 max_height=hp.V2S_MAX_HEIGHT,
                 max_width=hp.V2S_MAX_WIDTH,
                 tfr_ex_num=hp.V2S_TFR_EX_NUM):

        super().__init__(name,
                         w2v_model,
                         frames_path,
                         train_labels_filename,
                         eval_labels_filename,
                         prediction_labels_filename,
                         prediction_name,
                         model_base_dir,
                         sentence_max,
                         frames_max,
                         max_height,
                         max_width,
                         tfr_ex_num)

    def create_dataset(self, data_path, tfr_path, label_filename):
        """
        Creates input data set for Vid2Sentence which contains already preprocessed data:
        """

        with tf.gfile.GFile(label_filename, mode='r') as labels_csv:
            tfr_files = []

            if const.GCLOUD:
                labels = read_bucket_csv(label_filename)
            else:
                labels = pd.read_csv(labels_csv, delimiter=';', header=None)

            _, num_columns = labels.shape
            if num_columns == 1:
                cols = ['video_id']
            elif num_columns == 2:
                cols = ['video_id', 'sentence']
            else:
                raise ValueError('Csv file has an invalid amount of colums: {}'.format(num_columns))
            labels.columns = cols

            labels = labels[:256]  # TODO delete

            # logging variables
            sentence_lengths = []
            frame_lengths = []
            frame_widths = []
            frame_heights = []

            # get a generator for labels and features
            data_gen = self.generate_embedded_labels_frames(labels)
            model_path = const.INC_4_PATH
            tfr_writer = None
            with slim.arg_scope(models.inception_v4.inception_v4_arg_scope()):
                frames = tf.placeholder(tf.float32, shape=[None, None, None, 3])
                _, endpoints = models.inception_v4.inception_v4(frames, num_classes=1001, is_training=False,
                                                                create_aux_logits=False)
                layer = endpoints['Mixed_7d']

                with tf.Session() as sess:
                    saver = tf.train.Saver()
                    saver.restore(sess, model_path)

                    # for every video id in dataset list
                    for row in data_gen:
                        if num_columns == 1:
                            row_idx, video_id, encoded_frames = row
                            context = tf.train.Features(feature={'video_id': tfr_utils.int64_feature(video_id)})

                            # get last layer of CNN model
                            out_layers = sess.run(layer, feed_dict={frames: encoded_frames})

                            # convert to float32, reshape to one dimension and convert to string
                            out_layers = [np.reshape(ol.astype(np.float32), (-1)).tostring() for ol in out_layers]

                            feature_list = {
                                'out_layers': tfr_utils.bytes_feature_list(out_layers)
                            }

                        else:
                            row_idx, video_id, embedded_sentence, sentence_ids, encoded_frames = row
                            context = tf.train.Features(feature={'video_id': tfr_utils.int64_feature(video_id)})

                            # get data for logging
                            sentence_lengths.append(len(sentence_ids))
                            num, height, width, channels = encoded_frames.shape
                            frame_lengths.append(num)
                            frame_heights.append(height)
                            frame_widths.append(width)

                            # get last layer of CNN model
                            out_layers = sess.run(layer, feed_dict={frames: encoded_frames})

                            # convert to float32, reshape to one dimension and convert to string
                            out_layers = [np.reshape(ol.astype(np.float32), (-1)).tostring() for ol in out_layers]

                            feature_list = {
                                'label': tfr_utils.float_list_feature_list(embedded_sentence),
                                'label_ids': tfr_utils.float_list_feature_list(np.expand_dims(np.asarray(sentence_ids),
                                                                                     axis=-1)),
                                'out_layers': tfr_utils.bytes_feature_list(out_layers)
                            }

                        feature_lists = tf.train.FeatureLists(feature_list=feature_list)

                        example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

                        # create new tfr file every $tfr_ex_num examples
                        if row_idx % self.tfr_ex_num == 0:
                            tfr_file = tfr_path + '_{}.tfr'.format((int(row_idx / self.tfr_ex_num) + 1)
                                                                   * self.tfr_ex_num)
                            tfr_files.append(tfr_file)
                            if tfr_writer:
                                tfr_writer.close()
                                sys.stdout.flush()
                            tfr_writer = tf.python_io.TFRecordWriter(tfr_file)
                        tfr_writer.write(example.SerializeToString())

            if tfr_writer:
                tfr_writer.close()
                sys.stdout.flush()
            tf.reset_default_graph()
            if not const.GCLOUD and num_columns == 2:
                _save_histogram('Sentence Length', sentence_lengths, data_path)
                _save_histogram('Frame Length', frame_lengths, data_path)
                _save_histogram('Frame Width', frame_widths, data_path)
                _save_histogram('Frame Height', frame_heights, data_path)

            return tfr_files


class V2SOpticalFlowDataset(V2SBaseDataset):
    def __init__(self, name,
                 w2v_model,
                 frames_path,
                 train_labels_filename=None,
                 eval_labels_filename=None,
                 prediction_labels_filename=None,
                 prediction_name=None,
                 model_base_dir=const.VID2SENTENCE_DEFAULT_MODEL_PATH,
                 sentence_max=hp.V2S_SENTENCE_MAX,
                 frames_max=hp.V2S_FRAMES_MAX,
                 max_height=hp.V2S_MAX_HEIGHT,
                 max_width=hp.V2S_MAX_WIDTH,
                 tfr_ex_num=hp.V2S_TFR_EX_NUM):

        super().__init__(name,
                         w2v_model,
                         frames_path,
                         train_labels_filename,
                         eval_labels_filename,
                         prediction_labels_filename,
                         prediction_name,
                         model_base_dir,
                         sentence_max,
                         frames_max,
                         max_height,
                         max_width,
                         tfr_ex_num)

    def create_dataset(self, data_path, tfr_path, label_filename):
        """
        Creates input data set for Vid2Sentence which contains already preprocessed word data, jpg frames
        and optical flow frames for x and y direction:
        """
        with tf.gfile.GFile(label_filename, mode='r') as labels_csv:
            tfr_files = []

            if const.GCLOUD:
                labels = read_bucket_csv(label_filename)
            else:
                labels = pd.read_csv(labels_csv, delimiter=';', header=None)

            _, num_columns = labels.shape
            if num_columns == 1:
                cols = ['video_id']
            elif num_columns == 2:
                cols = ['video_id', 'sentence']
            else:
                raise ValueError('Csv file has an invalid amount of columns: {}'.format(num_columns))
            labels.columns = cols

            labels = labels[0:256]  # TODO delete

            # logging variables
            sentence_lengths = []
            frame_lengths = []
            frame_widths = []
            frame_heights = []

            # get a generator for labels and features
            data_gen = self.generate_embedded_labels_frames(labels, optical_flow=True)

            tfr_writer = None

            for row in data_gen:
                if num_columns == 1:

                    row_idx, video_id, encoded_frames, flows = row
                    # TODO ADAPT
                    # encode frames as jpg
                    jpg_frames = [tf.compat.as_bytes(image_processing.encode_jpg(img).tobytes()) for img in
                                  encoded_frames]
                    jpg_flow_x = [tf.compat.as_bytes(image_processing.encode_jpg(flow).tobytes()) for flow in
                                  flows[..., 0]]
                    jpg_flow_y = [tf.compat.as_bytes(image_processing.encode_jpg(flow).tobytes()) for flow in
                                  flows[..., 1]]

                    context = tf.train.Features(feature={'video_id': tfr_utils.int64_feature(video_id)})
                    feature_list = {
                        'out_layers': tfr_utils.bytes_feature_list(jpg_frames),
                        'flow_x': tfr_utils.bytes_feature_list(jpg_flow_x),
                        'flow_y': tfr_utils.bytes_feature_list(jpg_flow_y)
                    }


                else:
                    row_idx, video_id, embedded_sentence, sentence_ids, encoded_frames, flows = row

                    # get data for logging
                    sentence_lengths.append(len(sentence_ids))
                    num, height, width, channels = encoded_frames.shape
                    frame_lengths.append(num)
                    frame_heights.append(height)
                    frame_widths.append(width)

                    # encode frames as jpg
                    jpg_frames = [tf.compat.as_bytes(image_processing.encode_jpg(img).tobytes()) for img in
                                  encoded_frames]
                    jpg_flow_x = [tf.compat.as_bytes(image_processing.encode_jpg(flow).tobytes()) for flow in
                                  flows[..., 0]]
                    jpg_flow_y = [tf.compat.as_bytes(image_processing.encode_jpg(flow).tobytes()) for flow in
                                  flows[..., 1]]

                    context = tf.train.Features(feature={'video_id': tfr_utils.int64_feature(video_id)})
                    feature_list = {
                        'label': tfr_utils.float_list_feature_list(embedded_sentence),
                        'label_ids': tfr_utils.float_list_feature_list(np.expand_dims(np.asarray(sentence_ids), axis=-1)),
                        'out_layers': tfr_utils.bytes_feature_list(jpg_frames),
                        'flow_x': tfr_utils.bytes_feature_list(jpg_flow_x),
                        'flow_y': tfr_utils.bytes_feature_list(jpg_flow_y)
                    }

                feature_lists = tf.train.FeatureLists(feature_list=feature_list)

                example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

                # create new tfr file every $tfr_ex_num examples
                if row_idx % self.tfr_ex_num == 0:
                    tfr_file = tfr_path + '_{}.tfr'.format((int(row_idx / self.tfr_ex_num) + 1)
                                                           * self.tfr_ex_num)
                    tfr_files.append(tfr_file)
                    if tfr_writer:
                        tfr_writer.close()
                        sys.stdout.flush()
                    tfr_writer = tf.python_io.TFRecordWriter(tfr_file)
                tfr_writer.write(example.SerializeToString())

            if tfr_writer:
                tfr_writer.close()
                sys.stdout.flush()

            if not const.GCLOUD and not num_columns == 1:
                _save_histogram('Sentence Length', sentence_lengths, data_path)
                _save_histogram('Frame Length', frame_lengths, data_path)
                _save_histogram('Frame Width', frame_widths, data_path)
                _save_histogram('Frame Height', frame_heights, data_path)

            return tfr_files


def input_fn(tfrecord_filepath, batch_size, embedding_size, sentence_max, frames_max, is_training, frame_shape,
             input_data_type, bfloat16_mode, tpu=False, with_labels=True, seed=None):
    assert input_data_type in const.InputDataTypes, "Unknown input data type: {}".format(input_data_type)

    _file_pattern = os.path.join(tfrecord_filepath, "*.tfr")

    dataset = tf.data.Dataset.list_files(_file_pattern, shuffle=is_training)

    if is_training:
        dataset = dataset.repeat()

    # prefetch data files
    def _prefetch_dataset(_filename):
        _dataset = tf.data.TFRecordDataset(_filename).prefetch(1)
        return _dataset

    dataset = dataset.apply(tf.data.experimental.parallel_interleave(_prefetch_dataset, cycle_length=4, sloppy=is_training))

    # if is_training:
    dataset = dataset.shuffle(128, seed=seed)

    dataset = dataset.map(lambda x: _parse_jpg_tfr_data(x, embedding_size, input_data_type, with_labels, bfloat16_mode,
                                                        frame_shape), num_parallel_calls=hp.PARSER_PARALLEL_CALLS)
    dataset = dataset.prefetch(batch_size)

    h, w, c = frame_shape
    flow_h = h // hp.FLOW_RESIZE_FACTOR
    flow_w = w // hp.FLOW_RESIZE_FACTOR

    if bfloat16_mode:
        data_type = tf.bfloat16
    else:
        data_type = tf.float32

    if input_data_type == const.InputDataTypes.OpticalFlowVideoJpg:
        padded_shapes = {'video': [frames_max, h, w, c],
                         'video_id': [],
                         'flow_x': [frames_max, flow_h, flow_w, 1],
                         'flow_y': [frames_max, flow_h, flow_w, 1]}
        padding_values = {'video': tf.constant(0, dtype=data_type),
                          'video_id': tf.constant(0, dtype=tf.int64),
                          'flow_x': tf.constant(0, dtype=data_type),
                          'flow_y': tf.constant(0, dtype=data_type)}

        if with_labels:
            padded_shapes = (padded_shapes,
                             {'labels': [sentence_max, embedding_size],
                              'label_ids': [sentence_max, 1]})
            padding_values = (padding_values, {'labels': tf.constant(0, dtype=data_type),
                                               'label_ids': tf.constant(0, dtype=tf.int32)})
    else:
        padded_shapes = {'video': [frames_max, h, w, c],
                         'video_id': []}
        padding_values = {'video': tf.constant(0, dtype=data_type),
                          'video_id': tf.constant(0, dtype=tf.int64)}

        if with_labels:
            padded_shapes = (padded_shapes,
                             {'labels': [sentence_max, embedding_size],
                              'label_ids': [sentence_max, 1]})
            padding_values = (padding_values, {'labels': tf.constant(0, dtype=data_type),
                                               'label_ids': tf.constant(0, dtype=tf.int32)})

    # videos and sentences get padded with zeros up to their respective max size
    if tpu:
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values,
                                       drop_remainder=True)
    else:
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

    # AUTOTUNE allows the tf.data runtime to automatically tune the prefetch buffer sizes based on system and
    # environment
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def tpu_input_fn(tfrecord_filepath, embedding_size, sentence_max, frames_max, is_training, frame_shape,
                 input_data_type, bfloat16_mode, params, seed=None, with_labels=True):
    batch_size = params["batch_size"]
    return input_fn(tfrecord_filepath, batch_size, embedding_size, sentence_max, frames_max, is_training, frame_shape,
                    input_data_type, bfloat16_mode, tpu=True, seed=seed, with_labels=with_labels)


def get_pred_input_fn_and_bsize(label_filename, frames_path,input_data_type, frames_max, max_height,
                                max_width,  jpg_skip=1, opt_flow_bracket=None, opt_flow_frames_path=None, transfer_ckpt=None):
    assert input_data_type in const.InputDataTypes, "Unknown input data type: {}".format(input_data_type)
    optical_flow = input_data_type == const.InputDataTypes.OpticalFlowVideoJpg

    if const.GCLOUD:
        video_ids = read_bucket_csv(label_filename)
    else:
        video_ids = pd.read_csv(label_filename, delimiter=';', header=None)

    _, num_columns = video_ids.shape
    if num_columns == 1:
        cols = ['video_id']
    elif num_columns == 2:
        cols = ['video_id', 'sentence']
    else:
        raise ValueError('Csv file has an invalid amount of colums: {}'.format(num_columns))
    video_ids.columns = cols
    batch_size = video_ids.shape[0]
    videos = []

    def raw_input_fn():
        if input_data_type == const.InputDataTypes.PreprocessedVideoJpg:
            with slim.arg_scope(models.inception_v4.inception_v4_arg_scope()):
                frames = tf.placeholder(tf.float32, shape=[None, None, None, 3])
                _, endpoints = models.inception_v4.inception_v4(frames, num_classes=1001, is_training=False,
                                                                create_aux_logits=False)
                layer = endpoints['Mixed_7d']

                with tf.Session() as sess:
                    saver = tf.train.Saver()
                    saver.restore(sess, transfer_ckpt)

                    # for every video id in dataset list
                    for _, row in video_ids.iterrows():
                        encoded_frames = _get_frames(row['video_id'], frames_path, frames_max, max_height,
                                                     max_width)

                        # get last layer of CNN model
                        out_layers = sess.run(layer, feed_dict={frames: encoded_frames})

                        # convert to float32, reshape to one dimension and convert to string
                        videos.append(out_layers.astype(np.float32))

                    features = np.array(videos)

        elif input_data_type == const.InputDataTypes.OpticalFlowVideoJpg:
            flow_xs = []
            flow_ys = []
            for _, row in video_ids.iterrows():
                encoded_frames, flows = _get_frames_with_opt_flow(row['video_id'], jpg_skip,
                                                                  opt_flow_bracket,  frames_path,
                                                                  opt_flow_frames_path, frames_max,
                                                                  max_height, max_width, False)
                # TODO adapt for new opt flow
                videos.append(encoded_frames)
                flow_xs.append(flows[..., 0])
                flow_ys.append(flows[..., 1])
            flowarray_x = np.expand_dims(np.array(flow_xs), -1)
            flowarray_y = np.expand_dims(np.array(flow_ys), -1)
            features = {'video': np.array(videos), 'flow_x': np.array(flowarray_x), 'flow_y': np.array(flowarray_y)}

        else:
            for _, row in video_ids.iterrows():
                encoded_frames = _get_frames(row['video_id'], frames_path, frames_max, max_height,
                                             max_width)
                videos.append(encoded_frames)
            features = np.array(videos)

        dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset = dataset.batch(batch_size)
        return dataset

    return raw_input_fn, batch_size


def _parse_jpg_tfr_data(example, embedding_size, input_data_type, with_labels, bfloat16_mode, frame_shape=None):
    if input_data_type == const.InputDataTypes.PreprocessedVideoJpg:
        assert frame_shape is not None, "For a dataset with CNN output layers, their size must be set " \
                                            "in frame_shape attribute"
    context_features = {'video_id': tf.FixedLenFeature([], dtype=tf.int64)}

    sequence_features = {'out_layers': tf.FixedLenSequenceFeature([], dtype=tf.string)}
    label = None

    if bfloat16_mode:
        data_type = tf.bfloat16
    else:
        data_type = tf.float32

    if with_labels:
        sequence_features['label'] = tf.FixedLenSequenceFeature([embedding_size], dtype=tf.float32)
        sequence_features['label_ids'] = tf.FixedLenSequenceFeature([1], dtype=tf.float32)

    if input_data_type == const.InputDataTypes.OpticalFlowVideoJpg:
        sequence_features['flow_x'] = tf.FixedLenSequenceFeature([], dtype=tf.string)
        sequence_features['flow_y'] = tf.FixedLenSequenceFeature([], dtype=tf.string)

    features, sequence_features = tf.parse_single_sequence_example(example, context_features=context_features,
                                                                   sequence_features=sequence_features)
    if with_labels:
        label = {'labels': tf.cast(sequence_features['label'], dtype=data_type),
                 'label_ids': tf.to_int32(sequence_features['label_ids'])}

    if input_data_type == const.InputDataTypes.PreprocessedVideoJpg:
        video = tf.map_fn(lambda x: tf.reshape(tf.decode_raw(x, out_type=data_type), frame_shape),
                          sequence_features['out_layers'], dtype=data_type)
    else:
        video = tf.map_fn(lambda x: tf.cast(tf.image.decode_jpeg(x, channels=3), data_type),
                          sequence_features['out_layers'], dtype=data_type)

    if input_data_type == const.InputDataTypes.OpticalFlowVideoJpg:
        flow_x = tf.map_fn(lambda x: tf.cast(tf.image.decode_jpeg(x, channels=1), data_type),
                           sequence_features['flow_x'], dtype=data_type)
        flow_y = tf.map_fn(lambda x: tf.cast(tf.image.decode_jpeg(x, channels=1), data_type),
                           sequence_features['flow_y'], dtype=data_type)

        video_data = {'video': video, 'video_id': tf.cast(context_features['video_id'], tf.int64),
                      'flow_x': flow_x, 'flow_y': flow_y}

    else:
        video_id = tf.cast(features['video_id'], tf.int64)
        video_data = {'video': video, 'video_id': video_id}

    if with_labels:
        return video_data, label
    else:
        return video_data


def _get_frames(video_id, frames_path, frames_max, max_height, max_width):
    """
    Processes video frames.
    """
    frame_files = io_utils.get_frame_files(video_id, frames_path, frames_max)

    encoded_frames = []
    for file in frame_files:
        img = image_processing.load_image(file)
        prec_img = image_processing.crop_and_pad_image(max_height, max_width, img)
        encoded_frames.append(image_processing.shift_channels(prec_img))

    output = np.asarray(encoded_frames, dtype=np.float32)
    return output


def _get_frames_with_opt_flow(video_id, jpg_skip, opt_flow_bracket, frames_path, opt_path, frames_max, max_height,
                              max_width, training):
    """

    """
    jpg_frame_files, opt_brackets = io_utils.get_opt_flow_files(video_id, frames_path, opt_path, frames_max, jpg_skip,
                                                            opt_flow_bracket)
    bracket_ids, u_files, v_files = opt_brackets
    flip = False
    if training:
        flip = random.choice([True, False])
    enc_jpg_frames = []
    for frame in jpg_frame_files:
        img = image_processing.load_image(frame)
        proc_img = image_processing.crop_and_pad_image(max_height, max_width, img)
        if flip:
            proc_img = image_processing.flip_img(proc_img)
        enc_jpg_frames.append(image_processing.shift_channels(proc_img))
    jpg_output = np.asarray(enc_jpg_frames, dtype=np.float32)

    uv_enc_frames = []
    for u, v in zip(u_files, v_files):
        u_img = np.expand_dims(image_processing.load_opt_flow_image(u), -1)
        v_img = np.expand_dims(image_processing.load_opt_flow_image(v), -1)
        u_proc = image_processing.crop_and_pad_image(max_height // 2, max_width // 2, u_img)
        v_proc = image_processing.crop_and_pad_image(max_height // 2, max_width // 2, v_img)
        uv_proc = np.concatenate((u_proc, v_proc), -1)
        if flip:
            uv_proc = image_processing.flip_img(uv_proc)
        uv_enc_frames.append(uv_proc)

    uv_output = [np.asarray(enc, dtype=np.float32) for enc in uv_enc_frames]

    return jpg_output, uv_output, bracket_ids


def _save_histogram(name, array, model_path):
    plt.hist(array)
    plt.xlabel(name)
    plt.ylabel('Occurrence')
    plt.figtext(0.5, 0.7, 'Total: {} {}s. \n'
                          'Mean: {:.3} \n'
                          'Max: {}'
                .format(len(array),
                        name.split(' ')[0],
                        str(np.mean(array)),
                        str(np.amax(array))))
    path_name = str.lower(name).replace(' ', '_')
    plt.savefig(os.path.join(model_path, path_name + '_histogram'))
    plt.close()


def read_bucket_csv(gcs_path):
    file_stream = file_io.FileIO(gcs_path, mode='r')
    data = pd.read_csv(StringIO(file_stream.read()), delimiter=';', header=None)
    return data


def test_input_fn(params):
    batch_size = params["batch_size"]
    dataset1 = tf.data.Dataset.from_tensor_slices({"features": {"video_id": tf.ones([batch_size, 1]),
                                                                "video": {"video": tf.ones([batch_size, 8, 128, 128, 3], dtype=tf.bfloat16),
                                                                          "flow_stacks": tf.ones([batch_size, 8, 32, 32, 10], dtype=tf.bfloat16)
                                                                          },
                                                                },
                                                   "labels": {'label_ids': tf.ones([batch_size, 20, 1], dtype=tf.int64),
                                                              'labels': tf.ones([batch_size, 20, 512], dtype=tf.bfloat16)}
                                                   })
    dataset1 = dataset1.repeat()
    iterator = dataset1.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element['features'], next_element['labels']
