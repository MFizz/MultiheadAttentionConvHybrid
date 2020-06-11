import os
import pickle
import logging
import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import data.creation.word2vec
import log.log_utils
import constants.constants as const
import constants.hyper_params as hp


class Words2vec:
    """
            Creates a lookup table for words to vectors based on noise contrast estimation. See
            http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            for further information.
    """

    def __init__(self, name,
                 input_filename,
                 embedding_size=hp.W2V_EMBEDDING_SIZE,
                 neg_samples=hp.W2V_NEG_SAMPLES,
                 skip_word_window=hp.W2V_SKIP_WORD_WINDOW,
                 batch_size=hp.W2V_BATCH_SIZE,
                 epochs=hp.W2V_EPOCHS,
                 eval_frq=hp.W2V_EVAL_FRQ,
                 model_base_directory=const.WORD2VEC_DEFAULT_MODEL_PATH):
        """
            :param name: name of used input data
            :param embedding_size: size of embedding vector
            :param model_base_directory: base directory to save model files
            :param neg_samples: number of negative examples to sample for contrastive noise
            :param skip_word_window: number of context words to left and right of target word
            :param batch_size: number of words in one batch
            :param epochs: number of epochs to train
            :param eval_frq: how often evaluation is performed per run
        """
        model_directory = os.path.join(model_base_directory, name, 'model_emb_{}_negsamp_{}_skipwin_{}'.
                                       format(embedding_size, neg_samples, skip_word_window))
        # tensorflow bug in makedirs revert when fixed TODO
        tf.gfile.MakeDirs(model_directory)
        log.log_utils.add_words2vec_logging(os.path.join(model_directory, 'logs'))

        self.name = name
        self.input_filename = input_filename
        self.model_base_directory = model_base_directory
        self.embedding_size = embedding_size
        self.neg_samples = neg_samples
        self.skip_word_window = skip_word_window
        self.batch_size = batch_size
        self.epochs = epochs
        self.logger = logging.getLogger('main.words2vec')
        self.model_directory = model_directory
        self.eval_frq = eval_frq

        self.emb_lookup_table, self.vocabulary, self.word2id, self.data_set_size = self.get_model(input_filename)
        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))

    def __repr__(self):
        out_string = "Word2Vec Model {}, in {}, with embedding size {}, skip window {},\n" \
                     "{} negative samples for NCE-Loss, using {} as batch size with {} epochs."\
            .format(self.name,
                    self.model_directory,
                    self.embedding_size,
                    self.skip_word_window,
                    self.neg_samples,
                    self.batch_size,
                    self.epochs)
        return out_string

    def get_model(self, input_filename):
        """
        Gets trained word2vec model based on given params.
        :param input_filename: File name of input sentences in csv format.
        :return: output model file name containing the embedding lookup table
        """
        return self._get_or_create_model(input_filename)

    def _get_or_create_model(self, input_filename):
        """
        Looks for tfr_records and pickle files containing data set or creates it if not found.
        :param input_filename: File name of input sentences in csv format.
        :return: output model file name containing the embedding lookup table
        """

        # get or create data set, vocabulary, dictionary and word to id dictionary
        self.logger.info("Word2Vec Model Object created: {}".format(self))

        input_tfr_filename, vocabulary, word2id, dataset_size = \
            data.creation.word2vec.get_or_create_dataset(input_filename, self.model_base_directory,
                                                         self.skip_word_window)
        self.logger.info("Size of vocabulary: {}".format(len(vocabulary)))
        self.logger.info("Full data set size: {}".format(dataset_size))

        # create reverse dictionary
        id2word = dict(zip(word2id.values(), word2id.keys()))

        # see if final embedding is already computed
        model_output_filename = os.path.join(self.model_directory, 'final_embedding.p')
        try:
            embeddings = data.creation.word2vec.read_lookup_file(model_output_filename)
            self.logger.info('Model output embedding file found at: {}'.format(model_output_filename))

        except tf.errors.NotFoundError:
            self.logger.info('New embedding file is trained at: {}'.format(model_output_filename))
            vocabulary_size = len(vocabulary)

            # set the classifier
            classifier = tf.estimator.Estimator(
                model_fn=_word2vec_model_fn,
                model_dir=self.model_directory,
                params={'num_neg_samples': self.neg_samples,
                        'embedding_size': self.embedding_size,
                        'vocabulary_size': vocabulary_size
                        }
            )

            low_dim_embs = None
            img_path = os.path.join(self.model_directory, 'result_images')
            # tensorflow bug in makedirs revert when fixed TODO
            tf.gfile.MakeDirs(img_path)
            labels = [id2word[i] for i in range(vocabulary_size)]

            self.logger.info("Word2Vec step: 0 - examples: 0 - epoch: 0")

            if const.GCLOUD:
                embeddings = None
                for i in range(self.eval_frq):
                    epoch = (i + 1) * np.ceil(self.epochs / self.eval_frq)
                    step = epoch * dataset_size

                    # train the model
                    classifier.train(input_fn=lambda: _train_input_fn(
                        input_tfr_filename, self.batch_size, np.ceil(self.epochs / self.eval_frq)))

                    embeddings = _evaluate(vocabulary_size, classifier, id2word)

                    self.logger.info("Word2Vec step: {} - examples: {} - epoch: {}".format(
                        step / self.batch_size, step, epoch))

                final_embeddings_params = {'final_embeddings': embeddings}
                with tf.gfile.GFile(model_output_filename, mode='wb') as handle:
                    pickle.dump(final_embeddings_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                embeddings = _evaluate(vocabulary_size, classifier, id2word)
                start_emb = _plot_tsne(embeddings, labels, img_path, 0)

                for i in range(self.eval_frq):
                    epoch = (i + 1) * np.ceil(self.epochs / self.eval_frq)
                    step = epoch * dataset_size

                    # train the model
                    classifier.train(input_fn=lambda: _train_input_fn(
                        input_tfr_filename, self.batch_size, np.ceil(self.epochs / self.eval_frq)))

                    embeddings = _evaluate(vocabulary_size, classifier, id2word)
                    low_dim_embs = _plot_tsne(embeddings, labels, img_path, epoch)

                    self.logger.info("Word2Vec step: {} - examples: {} - epoch: {}".format(
                        step / self.batch_size, step, epoch))

                _plot_trace_with_labels(start_emb, low_dim_embs, labels, os.path.
                                        join(img_path, 'tsne_trace_epoch_{}.png'.format(self.epochs)))

                final_embeddings_params = {'final_embeddings': embeddings}
                with tf.gfile.GFile(model_output_filename, mode='wb') as handle:
                    pickle.dump(final_embeddings_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        tf.reset_default_graph()

        return embeddings, vocabulary, word2id, dataset_size


def _word2vec_model_fn(features, labels, mode, params):
    """
    Word to Vec model using noise contrast as loss. Explanation of the meaning of NCE loss:
    http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/ .
    :param params: Additional params dict: 'neg_samples': number of negative samples for nce loss.
    :param features: Features given by input_fn.
    :param labels: Labels given by input_fn.
    :param mode: tf.estimator.ModeKeys determining which mode to operate in (TRAIN, EVALUATE, PREDICT).
    :return:
    """
    # get parameters
    num_neg_samples = params['num_neg_samples']
    embed_size = params['embedding_size']
    vocab_size = params['vocabulary_size']

    # initialize random embedding vectors and embed input
    initial_width = 0.5 / embed_size
    embeddings = tf.Variable(
        tf.random_uniform([vocab_size, embed_size], -initial_width, initial_width), name='embedding')

    if mode == tf.estimator.ModeKeys.PREDICT:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

        predictions = {
            'final_embedding': normalized_embeddings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # get current embedding for input
    embed = tf.nn.embedding_lookup(embeddings, features)

    # initialize weights and biases for noise contrast estimation
    nce_weights = tf.Variable(
        tf.truncated_normal([vocab_size, embed_size],
                            stddev=1.0 / np.sqrt(embed_size)))
    nce_biases = tf.Variable(tf.zeros([vocab_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative ws each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=tf.reshape(labels, [-1, 1]),
                       inputs=embed,
                       num_sampled=num_neg_samples,
                       num_classes=vocab_size))

    # create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # doing gradient descent
    train_op = tf.train.GradientDescentOptimizer(1.0).minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def _evaluate(vocabulary_size, classifier, id2word):
    logger = logging.getLogger('main.words2vec')
    num_eval_words = 20
    max_eval_id = vocabulary_size
    valid_examples = list(np.random.choice(max_eval_id, num_eval_words, replace=False))
    valid_data = {'eval_words': valid_examples}

    # get predictions
    predicted_embeddings = classifier.predict(
        input_fn=lambda: _eval_input_fn(valid_data, labels=None, eval_batch_size=num_eval_words))
    final_embedding = []
    for i, embedding in enumerate(predicted_embeddings):
        final_embedding.append(embedding['final_embedding'])
        if i == vocabulary_size - 1:
            break

    embedding_array = np.array(final_embedding)
    valid_embedded = embedding_array[valid_examples]

    similarity = np.matmul(valid_embedded, embedding_array.T)
    for i in range(num_eval_words):
        valid_word = id2word[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-similarity[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = id2word[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        logger.debug(log_str)

    return embedding_array


def _parse_tfr_data(example):
    """
    Parser for tfr examples written in write_to_tfr().
    :param example: tfrecord example
    :return: example in readable form
    """
    keys_to_features = {'word': tf.FixedLenFeature((), tf.int64, default_value=0),
                        'label': tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example, keys_to_features)

    return parsed_features['word'], parsed_features['label']


def _plot_with_labels(low_dim_embs, labels, filename):
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


def _plot_trace_with_labels(start_emb, end_emb, labels, filename):
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = end_emb[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    for i in range(len(labels)):
        x, y = start_emb[i, :]
        new_x, new_y = end_emb[i, :]
        plt.plot([x, new_x], [y, new_y], color='r')
    plt.savefig(filename)


def _train_input_fn(tfrecord_filename, train_batch_size, train_epochs, seed=None):
    """
    Input function for training. Does batching, shuffeling and epoching for the data set.
    :param tfrecord_filename: the tfrecord file containing the data set
    :param train_batch_size: number of target words in batch
    :param train_epochs: number of epochs trained
    :return: next element in batch
    """
    data_set = tf.data.TFRecordDataset(tfrecord_filename)
    data_set = data_set.map(_parse_tfr_data)
    shuffeled_data_set = data_set.shuffle(1000, seed=seed)
    epochd_data_set = shuffeled_data_set.repeat(train_epochs)
    batched_dataset = epochd_data_set.batch(train_batch_size)
    return batched_dataset


def _eval_input_fn(features, labels, eval_batch_size):
    """
    Input function for evaluation or prediction
    """
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    data_set = tf.data.Dataset.from_tensor_slices(inputs)
    data_set = data_set.batch(eval_batch_size)
    return data_set


def _plot_tsne(embeddings, labels, img_path, epoch):
    """
    Plots given embedding into 3 dimensions
    :param embeddings: word embedding array
    :param labels:
    :param img_path:
    :param epoch:
    :return:
    """
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(embeddings)
    _plot_with_labels(low_dim_embs, labels, os.path.join(img_path, 'tsne_epoch_{}.png'.format(epoch)))

    return low_dim_embs
