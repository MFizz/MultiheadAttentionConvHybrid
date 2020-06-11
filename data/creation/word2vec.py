import os
import pickle
import logging
import tensorflow as tf

import data.creation.word_processing as word_procession


def get_or_create_dataset(input_filename, model_base_directory, skip_window):
    """
    Looks for input data set for word2vec depending on skip_window or creates it.
    :param input_filename: file name of input sentences in csv format
    :param model_base_directory: base directory to save model files
    :param skip_window: Size of context word window
    :return: data set filename, meta data file name, list of unique words, dictionary from word to id,
    number of examples in data set
    """
    logger = logging.getLogger('main.words2vec')

    # get file names
    input_base_filename = os.path.join(model_base_directory,  'train_skip-win_{}'.format(skip_window))
    input_tfr_filename = input_base_filename + '.tfrecord'
    input_pickle_filename = input_base_filename + '.p'

    try:
        with tf.gfile.GFile(input_pickle_filename, mode='rb') as handle:
            saved_params = pickle.load(handle)
        unique_words = saved_params['unique_words']
        dictionary = saved_params['dictionary']
        train_example_size = saved_params['train_example_size']

        logger.info('Word2Vec model meta file used: {}'.format(input_pickle_filename))

        # check if tfrecord exists
        if not tf.gfile.Exists(input_tfr_filename):
            logger.info('No Word2Vec model input files found, creating file at: {}'.format(input_tfr_filename))
            sentences = word_procession.read_csv_label_sentences(input_filename)
            # write training data to tfr format
            train_example_size = _write_to_tfr(sentences, input_tfr_filename, skip_window, dictionary)
        else:
            logger.info('Word2Vec model input file used: {}'.format(input_tfr_filename))

    except tf.errors.NotFoundError:
        logger.info('No Word2Vec model meta and input files found, creating files at \n'
                    'meta: {} \n'
                    'input: {}'.format(input_pickle_filename, input_tfr_filename))
        # get vocabulary
        sentences = word_procession.read_csv_label_sentences(input_filename)
        unique_words = word_procession.create_vocabulary(sentences)

        # create dictionary from word to id and put unknown for id 0
        dictionary = dict(zip(unique_words, range(1, 1 + len(unique_words))))
        dictionary[u"unk"] = 0
        unique_words.add(u"unk")

        # write training data to tfr format for reusability
        train_example_size = _write_to_tfr(sentences, input_tfr_filename, skip_window, dictionary)

        word2vec_params = {'unique_words': unique_words, 'dictionary': dictionary,
                           'train_example_size': train_example_size}
        with tf.gfile.GFile(input_pickle_filename, mode='wb') as handle:
            pickle.dump(word2vec_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return input_tfr_filename, unique_words, dictionary, train_example_size


def read_lookup_file(lookup_filename):
    """
    Read lookup table from pickle file
    :param lookup_filename: file name of pickle file containing embedding lookup table
    :return:
    """
    with tf.gfile.GFile(lookup_filename, mode='rb') as handle:
        saved_params = pickle.load(handle)
    return saved_params['final_embeddings']


def _write_to_tfr(sentence_list, train_filename, skip_window, dictionary):
    """
    Writes lists of sentences to training examples in the form of "(target_word_id, context_word_id)".
    :param dictionary: Dictionary form word to id
    :param skip_window: Size of window around target words to be regarded as context words.
    :param sentence_list: List of target sentences.
    :param train_filename: File name to write tfr data set to.
    :return: Number of written examples.
    """
    splits_list = [sentence.split(" ") for sentence in sentence_list]
    lowercase_splits_list = [[str.lower(word) for word in sentence] for sentence in splits_list]
    encoded_splits_list = [[dictionary[word] for word in sentence] for sentence in lowercase_splits_list]
    example_size = 0
    with tf.python_io.TFRecordWriter(train_filename) as writer:
        for sentence in encoded_splits_list:
            for i, word in enumerate(sentence):
                context_words = word_procession.get_context_words(i, skip_window, sentence)
                for c_word in context_words:
                    # one example consists of the input word and one of the context words around it
                    feature = {'word': tf.train.Feature(int64_list=tf.train.Int64List(value=[word])),
                               'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[c_word]))}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    example_size += 1
    return example_size
