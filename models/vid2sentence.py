import os
import logging
import tensorflow as tf

import log.log_utils
import data.creation.vid2sentence
import models.vid2sentence_utils
import models.vid2sentence_modules
import models.words2vec
import constants.constants as const
import constants.hyper_params as hp
import models.hooks


class Vid2SentenceBase:

    def __init__(self, name,
                 learning_rates,
                 dataset,
                 model_fn,
                 encoder_type,
                 normalization_fn,
                 activation,
                 video_emb_type,
                 model_base_dir,
                 frame_shape,
                 batch_size,
                 epochs_per_eval,
                 dropout_rate,
                 layer_stack,
                 mult_heads,
                 ff_hidden_units,
                 bfloat16_mode,
                 optimizer_type,
                 adam_beta1,
                 adam_beta2,
                 adam_epsilon,
                 adafactor_decay_rate,
                 adafactor_beta1,
                 adafactor_epsilon1,
                 adafactor_epsilon2,
                 label_smoothing,
                 weight_mode,
                 max_n_gram,
                 delta_step,
                 min_step,
                 delta_value,
                 warm_start_dir=None):

        model_directory = os.path.join(model_base_dir, name, 'model_emb_{}_negsamp_{}_skipwin_{}'.
                                       format(dataset.w2v_model.embedding_size, dataset.w2v_model.neg_samples,
                                              dataset.w2v_model.skip_word_window))
        # tensorflow bug in makedirs revert when fixed TODO
        tf.gfile.MakeDirs(model_directory)
        log.log_utils.add_vid2sentence_logging(os.path.join(model_directory, 'logs'))
        self.logger = logging.getLogger('main.vid2sentence')

        self.dataset = dataset

        # model
        self.name = name
        self.model_dir = model_directory
        self.model_fn = model_fn
        self.bfloat16_mode = bfloat16_mode

        # hyper params
        self.enc_type = encoder_type
        self.learning_rates = learning_rates
        self.batch_size = batch_size
        self.epochs_per_eval = epochs_per_eval
        self.dropout_rate = dropout_rate
        self.layer_stack = layer_stack
        self.mult_heads = mult_heads
        self.ff_hidden_units = ff_hidden_units
        self.normalization_fn = normalization_fn
        self.activation = activation
        self.video_emb_type = video_emb_type
        self.warm_start_dir = warm_start_dir
        self.frame_shape = frame_shape
        self.optimizer_type = optimizer_type

        # optimizer params - adam
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.adafactor_decay_rate = adafactor_decay_rate
        self.adafactor_beta1 = adafactor_beta1
        self.adafactor_epsilon1 = adafactor_epsilon1
        self.adafactor_epsilon2 = adafactor_epsilon2
        self.label_smoothing = label_smoothing
        self.weight_mode = weight_mode

        # eval metric params
        self.max_n_gram = max_n_gram

        # early stopping params
        self.delta_step = delta_step
        self.min_step = min_step
        self.delta_value = delta_value

        # get default values
        if self.video_emb_type is None:
            assert dataset, "If no dataset is given, video_emb_type must be specified"

            if self.dataset.input_data_type() == const.InputDataTypes.VideoJpg:
                self.video_emb_type = const.VideoEmbeddingType.InceptionV4

        if frame_shape is None:
            assert dataset, "If no dataset is given, frame_shape must be specified"

            if self.dataset.input_data_type() == const.InputDataTypes.PreprocessedVideoJpg:
                self.frame_shape = hp.CNN_MODEL_SHAPE
                self.warm_start_dir = None

            else:
                self.frame_shape = hp.FRAME_SHAPE
                if self.dataset.input_data_type() == const.InputDataTypes.OpticalFlowVideoJpg:
                    self.warm_start_dir = None
        else:
            self.frame_shape = frame_shape

    def __repr__(self):
        out_string = "Vid2Sentence Model {}, in {}\n" \
                     "maximal sentence length {}\n" \
                     "maximal video length {} frames\n" \
                     "frames of height {} and width {}\n" \
                     "batch_size {}\n" \
                     "{} epochs per evaluation\n" \
                     "Word2Vec Model is: {}" \
            .format(self.name, self.model_dir,
                    self.dataset.sentence_max,
                    self.dataset.frames_max,
                    self.dataset.max_height, self.dataset.max_width,
                    self.batch_size,
                    self.epochs_per_eval,
                    self.dataset.w2v_model)

        return out_string


class Vid2Sentence(Vid2SentenceBase):

    def __init__(self, name,
                 learning_rates,
                 dataset,
                 encoder_type,
                 video_emb_type=None,
                 normalization_fn=models.vid2sentence_utils.layer_normalization,
                 activation=hp.V2S_ACTIVATION,
                 model_base_dir=const.VID2SENTENCE_DEFAULT_MODEL_PATH,
                 frame_shape=None,
                 batch_size=hp.V2S_BATCH_SIZE,
                 epochs_per_eval=hp.V2S_EPOCHS_PER_EVAL,
                 dropout_rate=hp.V2S_DROPOUT_RATE,
                 layer_stack=hp.V2S_LAYERS,
                 mult_heads=hp.V2S_NUM_MULT_HEADS,
                 ff_hidden_units=hp.V2S_FF_HIDDEN_UNITS,
                 bfloat16_mode=hp.V2S_BFLOAT16_MODE,
                 optimizer_type=hp.V2S_OPTIMIZER_TYPE,
                 adam_beta1=hp.V2S_ADAM_BETA1,
                 adam_beta2=hp.V2S_ADAM_BETA2,
                 adam_epsilon=hp.V2S_ADAM_EPSILON,
                 adafactor_decay_rate=hp.V2S_ADAFACTOR_DECAY,
                 adafactor_beta1=hp.V2S_ADAFACTOR_BETA,
                 adafactor_epsilon1=hp.V2S_ADAFACTOR_EPSILON1,
                 adafactor_epsilon2=hp.V2S_ADAFACTOR_EPSILON2,
                 label_smoothing=hp.V2S_LBL_SMOOTHING,
                 weight_mode=hp.V2S_WEIGHT_MODE,
                 max_n_gram=hp.V2S_MAX_N_GRAM,
                 delta_step=hp.V2S_DELTA_STEP,
                 min_step=hp.V2S_MIN_STEP,
                 delta_value=hp.V2S_DELTA_VAL,
                 warm_start_dir=const.INC_4_PATH):

        super().__init__(name,
                         learning_rates,
                         dataset,
                         models.vid2sentence_modules.vid2sentence,
                         encoder_type,
                         normalization_fn,
                         activation,
                         video_emb_type,
                         model_base_dir,
                         frame_shape,
                         batch_size,
                         epochs_per_eval,
                         dropout_rate,
                         layer_stack,
                         mult_heads,
                         ff_hidden_units,
                         bfloat16_mode,
                         optimizer_type,
                         adam_beta1,
                         adam_beta2,
                         adam_epsilon,
                         adafactor_decay_rate,
                         adafactor_beta1,
                         adafactor_epsilon1,
                         adafactor_epsilon2,
                         label_smoothing,
                         weight_mode,
                         max_n_gram,
                         delta_step,
                         min_step,
                         delta_value,
                         warm_start_dir)

    def fixed_epoch_train_eval_loop(self, epochs):
        self.logger.info("Start train and eval loop (fixed epochs - {}) for: Vid2Sentence Model Object: {}".
                         format(epochs, self))
        return self.train_eval_loop(epochs)

    def early_stopping_train_eval_loop(self):
        self.logger.info("Start train and eval loop (early stopping) for: Vid2Sentence Model Object: {}".
                         format(self))
        return self.train_eval_loop()

    def train_eval_loop(self, epochs=None):

        def train_input_fn():
            return data.creation.vid2sentence.input_fn(tfrecord_filepath=self.dataset.train_data_path,
                                                       batch_size=self.batch_size,
                                                       embedding_size=self.dataset.w2v_model.embedding_size,
                                                       sentence_max=self.dataset.sentence_max,
                                                       frames_max=self.dataset.frames_max,
                                                       is_training=True,
                                                       frame_shape=self.frame_shape,
                                                       input_data_type=self.dataset.input_data_type(),
                                                       bfloat16_mode=self.bfloat16_mode)

        def eval_input_fn():
            return data.creation.vid2sentence.input_fn(tfrecord_filepath=self.dataset.eval_data_path,
                                                       batch_size=self.batch_size,
                                                       embedding_size=self.dataset.w2v_model.embedding_size,
                                                       sentence_max=self.dataset.sentence_max,
                                                       frames_max=self.dataset.frames_max,
                                                       is_training=False,
                                                       frame_shape=self.frame_shape,
                                                       input_data_type=self.dataset.input_data_type(),
                                                       bfloat16_mode=self.bfloat16_mode)

        start_steps = []
        end_steps = []
        # create models for each given learning rate
        for lr in self.learning_rates:
            ext_dir = "{}_encoder".format(self.enc_type.name)
            if self.video_emb_type:
                ext_dir = os.path.join(ext_dir, "{}_video_emb".format(self.video_emb_type.name))
            lr_model_dir = os.path.join(self.model_dir, ext_dir, 'lr_{}'.format(lr))
            log.log_utils.add_lr_logging(os.path.join(lr_model_dir, 'logs'))

            lr_logger = logging.getLogger('main.vid2sentence.lr')
            lr_logger.info("Starting Model: {} \n"
                           "with learning rate {}".format(self, lr))

            ws = None
            vars_to_start = None
            if self.video_emb_type == const.VideoEmbeddingType.InceptionV4:
                vars_to_start = '.*InceptionV4.*'
            elif self.video_emb_type == const.VideoEmbeddingType.MobilenetV1:
                vars_to_start = '.*MobilenetV1.*'
            if self.warm_start_dir:
                ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=self.warm_start_dir,
                                                    vars_to_warm_start=vars_to_start)

            # set the classifier
            classifier = tf.estimator.Estimator(
                model_fn=self.model_fn,
                model_dir=lr_model_dir,
                params={'dropout_rate': self.dropout_rate,
                        'num_heads': self.mult_heads,
                        'feed_forward_hidden_units': self.ff_hidden_units,
                        'normalization_fn': self.normalization_fn,
                        'activation': self.activation,
                        'video_emb_type': self.video_emb_type,
                        'layer_stack': self.layer_stack,
                        'num_vocab': len(self.dataset.w2v_model.vocabulary),
                        'learning_rate': lr,
                        'bfloat16_mode': self.bfloat16_mode,
                        'optimizer_type': self.optimizer_type,
                        'adam_beta1': self.adam_beta1,
                        'adam_beta2': self.adam_beta2,
                        'adam_epsilon': self.adam_epsilon,
                        'adafactor_decay_rate': self.adafactor_decay_rate,
                        'adafactor_beta1': self.adafactor_beta1,
                        'adafactor_epsilon1': self.adafactor_epsilon1,
                        'adafactor_epsilon2': self.adafactor_epsilon2,
                        'label_smoothing': self.label_smoothing,
                        'weight_mode': self.weight_mode,
                        'max_n_gram': self.max_n_gram,
                        'input_data_type': self.dataset.input_data_type(),
                        'enc_type': self.enc_type,
                        'tpu': False,
                        },
                warm_start_from=ws
            )

            best_loss = float("inf")
            best_step = 0
            hooks = list()
            hooks.append(tf.train.ProfilerHook(1, output_dir=lr_model_dir))

            if const.GCLOUD:
                classifier.train(input_fn=train_input_fn, hooks=hooks)

            start_steps.append(classifier.get_variable_value("global_step"))
            cur_epoch = 0
            while True:
                cur_epoch += 1
                if epochs and epochs < cur_epoch:
                    break
                try:
                    eval_dict = classifier.evaluate(input_fn=eval_input_fn)

                    # stop early if metrics are not improving
                    lr_logger.info("Global step: {}".format(eval_dict['global_step']))
                    if eval_dict['global_step'] >= self.min_step:
                        if eval_dict['loss'] < best_loss - self.delta_value:
                            best_loss = eval_dict['loss']
                            best_step = eval_dict['global_step']
                        if best_step < eval_dict['global_step'] - self.delta_step:
                            lr_logger.info(
                                "Lr: {}. Early stopping bc best loss {} and best step {} < {}. cur loss = {}".format(
                                    lr,
                                    best_loss,
                                    best_step,
                                    eval_dict['global_step'],
                                    eval_dict['loss']))
                            break
                        else:
                            lr_logger.info("Lr: {}. Not stopping early bc best loss {} and best step {} >= {}. cur "
                                           "loss = {}".format(lr, best_loss, best_step, eval_dict['global_step'],
                                                              eval_dict['loss']))

                except tf.train.NanLossDuringTrainingError:
                    lr_logger.warning("NaN error with lr {}".format(lr))

            classifier.train(input_fn=train_input_fn, hooks=hooks)
            end_steps.append(classifier.get_variable_value("global_step"))
        return start_steps, end_steps, self.learning_rates

    def eval_only(self, model_dir):
        self.logger.info("Start evaluation for: Vid2Sentence Model Object: {}".format(self))

        def eval_input_fn():
            return data.creation.vid2sentence.input_fn(tfrecord_filepath=self.dataset.eval_data_path,
                                                       batch_size=self.batch_size,
                                                       embedding_size=self.dataset.w2v_model.embedding_size,
                                                       sentence_max=self.dataset.sentence_max,
                                                       frames_max=self.dataset.frames_max,
                                                       is_training=False,
                                                       frame_shape=self.frame_shape,
                                                       input_data_type=self.dataset.input_data_type(),
                                                       bfloat16_mode=self.bfloat16_mode)

        # set the classifier
        classifier = tf.estimator.Estimator(
            model_fn=self.model_fn,
            model_dir=model_dir,
            params={'dropout_rate': self.dropout_rate,
                    'num_heads': self.mult_heads,
                    'feed_forward_hidden_units': self.ff_hidden_units,
                    'normalization_fn': self.normalization_fn,
                    'activation': self.activation,
                    'video_emb_type': self.video_emb_type,
                    'layer_stack': self.layer_stack,
                    'num_vocab': len(self.dataset.w2v_model.vocabulary),
                    'max_n_gram': self.max_n_gram,
                    'input_data_type': self.dataset.input_data_type(),
                    'enc_type': self.enc_type,
                    'tpu': False,
                    },
        )

        classifier.evaluate(input_fn=eval_input_fn)

    def predict(self, input_files, model_dir, beam_size, with_labels=False, transfer_ckpt=None, array_input=False):

        self.logger.info("Start prediction with: Vid2Sentence Model Object: {}".format(self))
        if not array_input:
            dataset = input_files

            def input_fn():
                return data.creation.vid2sentence.input_fn(tfrecord_filepath=dataset.predict_data_path,
                                                           batch_size=1,  # TODO
                                                           embedding_size=dataset.w2v_model.embedding_size,
                                                           sentence_max=dataset.sentence_max,
                                                           frames_max=dataset.frames_max,
                                                           is_training=False,
                                                           frame_shape=self.frame_shape,
                                                           input_data_type=dataset.input_data_type(),
                                                           bfloat16_mode=self.bfloat16_mode,
                                                           with_labels=with_labels)
        else:
            input_fn, batch_size = \
                data.creation.vid2sentence.get_pred_input_fn_and_bsize(input_files,
                                                                       self.dataset.frames_path,
                                                                       self.dataset.input_data_type(),
                                                                       self.dataset.frames_max,
                                                                       self.dataset.max_height,
                                                                       self.dataset.max_width,
                                                                       transfer_ckpt)
        # set the classifier
        classifier = tf.estimator.Estimator(
            model_fn=self.model_fn,
            model_dir=model_dir,
            params={'dropout_rate': self.dropout_rate,
                    'num_heads': self.mult_heads,
                    'feed_forward_hidden_units': self.ff_hidden_units,
                    'normalization_fn': self.normalization_fn,
                    'activation': self.activation,
                    'video_emb_type': self.video_emb_type,
                    'layer_stack': self.layer_stack,
                    'num_vocab': len(self.dataset.w2v_model.vocabulary),
                    'bfloat16_mode': self.bfloat16_mode,
                    'input_data_type': self.dataset.input_data_type(),
                    'enc_type': self.enc_type,
                    'tpu': False,
                    'pred_batch_size': 1,  # TODO
                    'pred_label_emb_size': self.dataset.w2v_model.embedding_size,
                    'pred_max_sentence': self.dataset.sentence_max,
                    'emb_lookup_table': self.dataset.w2v_model.emb_lookup_table,
                    'with_labels': with_labels,
                    'word2id': self.dataset.w2v_model.word2id,
                    'beam_size': beam_size,
                    'alpha': hp.ALPHA
                    },
        )

        id2word = self.dataset.w2v_model.id2word
        predictions = classifier.predict(input_fn)
        pred_sentences = []
        pred_probs = []
        for pred in predictions:
            pred_labels = pred['labels']
            words = []
            for labels in pred_labels:
                words.append([id2word[w_id] for w_id in labels])
            pred_sentences.append([id2word[w_id] for w_id in pred_labels])
            pred_probs.append(pred['probs'])

        return pred_sentences, pred_probs


class Vid2SentenceTPU(Vid2SentenceBase):

    def __init__(self, name,
                 learning_rates,
                 dataset,
                 encoder_type,
                 video_emb_type=None,
                 normalization_fn=models.vid2sentence_utils.layer_normalization,
                 activation=hp.V2S_ACTIVATION,
                 model_base_dir=const.VID2SENTENCE_DEFAULT_MODEL_PATH,
                 frame_shape=None,
                 batch_size=hp.V2S_BATCH_SIZE,
                 epochs_per_eval=hp.V2S_EPOCHS_PER_EVAL,
                 dropout_rate=hp.V2S_DROPOUT_RATE,
                 layer_stack=hp.V2S_LAYERS,
                 mult_heads=hp.V2S_NUM_MULT_HEADS,
                 ff_hidden_units=hp.V2S_FF_HIDDEN_UNITS,
                 bfloat16_mode=hp.V2S_BFLOAT16_MODE,
                 optimizer_type=hp.V2S_OPTIMIZER_TYPE,
                 adam_beta1=hp.V2S_ADAM_BETA1,
                 adam_beta2=hp.V2S_ADAM_BETA2,
                 adam_epsilon=hp.V2S_ADAM_EPSILON,
                 adafactor_decay_rate=hp.V2S_ADAFACTOR_DECAY,
                 adafactor_beta1=hp.V2S_ADAFACTOR_BETA,
                 adafactor_epsilon1=hp.V2S_ADAFACTOR_EPSILON1,
                 adafactor_epsilon2=hp.V2S_ADAFACTOR_EPSILON2,
                 label_smoothing=hp.V2S_LBL_SMOOTHING,
                 weight_mode=hp.V2S_WEIGHT_MODE,
                 max_n_gram=hp.V2S_MAX_N_GRAM,
                 delta_step=hp.V2S_DELTA_STEP,
                 min_step=hp.V2S_MIN_STEP,
                 delta_value=hp.V2S_DELTA_VAL,
                 warm_start_dir=const.INC_4_PATH):

        super().__init__(name,
                         learning_rates,
                         dataset,
                         models.vid2sentence_modules.vid2sentence,
                         encoder_type,
                         normalization_fn,
                         activation,
                         video_emb_type,
                         model_base_dir,
                         frame_shape,
                         batch_size,
                         epochs_per_eval,
                         dropout_rate,
                         layer_stack,
                         mult_heads,
                         ff_hidden_units,
                         bfloat16_mode,
                         optimizer_type,
                         adam_beta1,
                         adam_beta2,
                         adam_epsilon,
                         adafactor_decay_rate,
                         adafactor_beta1,
                         adafactor_epsilon1,
                         adafactor_epsilon2,
                         label_smoothing,
                         weight_mode,
                         max_n_gram,
                         delta_step,
                         min_step,
                         delta_value,
                         warm_start_dir)

    def fixed_expoch_train_eval_loop(self,
                                     use_tpu,
                                     tpu_iterations=hp.TPU_ITERATIONS,
                                     tpu_shards=hp.TPU_NUM_SHARDS,
                                     tpu_zone=hp.TPU_ZONE,
                                     gcp_tpu_name=hp.TPU_GCP_NAME,
                                     tpu_name=hp.TPU_NAME,
                                     train_batch_size=hp.TPU_TRAIN_BATCH_SIZE,
                                     eval_batch_size=hp.TPU_EVAL_BATCH_SIZE,
                                     train_examples_per_epoch=hp.TPU_TRAIN_EXAMPLES_PER_EPOCH,
                                     eval_examples_per_epoch=hp.TPU_EVAL_EXAMPLES_PER_EPOCH,
                                     epochs=hp.TPU_EPOCHS):
        self.logger.info("Start train and eval loop (fixed epochs - {}) for: Vid2Sentence Model Object: {}".
                         format(epochs, self))

        return self.train_eval_loop(use_tpu,
                                    tpu_iterations,
                                    tpu_shards,
                                    tpu_zone,
                                    gcp_tpu_name,
                                    tpu_name,
                                    train_batch_size,
                                    eval_batch_size,
                                    train_examples_per_epoch,
                                    eval_examples_per_epoch,
                                    epochs)

    def early_stopping_train_eval_loop(self,
                                       use_tpu,
                                       tpu_iterations=hp.TPU_ITERATIONS,
                                       tpu_shards=hp.TPU_NUM_SHARDS,
                                       tpu_zone=hp.TPU_ZONE,
                                       gcp_tpu_name=hp.TPU_GCP_NAME,
                                       tpu_name=hp.TPU_NAME,
                                       train_batch_size=hp.TPU_TRAIN_BATCH_SIZE,
                                       eval_batch_size=hp.TPU_EVAL_BATCH_SIZE,
                                       train_examples_per_epoch=hp.TPU_TRAIN_EXAMPLES_PER_EPOCH,
                                       eval_examples_per_epoch=hp.TPU_EVAL_EXAMPLES_PER_EPOCH):
        self.logger.info("Start train and eval loop (early stopping) for: Vid2Sentence Model Object: {}".format(self))

        return self.train_eval_loop(use_tpu,
                                    tpu_iterations,
                                    tpu_shards,
                                    tpu_zone,
                                    gcp_tpu_name,
                                    tpu_name,
                                    train_batch_size,
                                    eval_batch_size,
                                    train_examples_per_epoch,
                                    eval_examples_per_epoch)

    def train_eval_loop(self,
                        use_tpu,
                        tpu_iterations,
                        tpu_shards,
                        tpu_zone,
                        gcp_tpu_name,
                        tpu_name,
                        train_batch_size,
                        eval_batch_size,
                        train_examples_per_epoch,
                        eval_examples_per_epoch,
                        epochs=None):

        def train_input_fn(params):
            return data.creation.vid2sentence.tpu_input_fn(tfrecord_filepath=self.dataset.train_data_path,
                                                           embedding_size=self.dataset.w2v_model.embedding_size,
                                                           sentence_max=self.dataset.sentence_max,
                                                           frames_max=self.dataset.frames_max,
                                                           is_training=True,
                                                           frame_shape=self.frame_shape,
                                                           input_data_type=self.dataset.input_data_type(),
                                                           bfloat16_mode=self.bfloat16_mode,
                                                           params=params)

        def eval_input_fn(params):
            return data.creation.vid2sentence.tpu_input_fn(tfrecord_filepath=self.dataset.eval_data_path,
                                                           embedding_size=self.dataset.w2v_model.embedding_size,
                                                           sentence_max=self.dataset.sentence_max,
                                                           frames_max=self.dataset.frames_max,
                                                           is_training=False,
                                                           frame_shape=self.frame_shape,
                                                           input_data_type=self.dataset.input_data_type(),
                                                           bfloat16_mode=self.bfloat16_mode,
                                                           params=params)

        ws = None
        vars_to_start = None
        if self.video_emb_type == const.VideoEmbeddingType.InceptionV4:
            vars_to_start = '.*InceptionV4.*'
        elif self.video_emb_type == const.VideoEmbeddingType.MobilenetV1:
            vars_to_start = '.*MobilenetV1.*'
        if self.warm_start_dir:
            ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=self.warm_start_dir,
                                                vars_to_warm_start=vars_to_start)

        if use_tpu:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                tpu_name,
                zone=tpu_zone,
                project=gcp_tpu_name)
            tpu_grpc_url = tpu_cluster_resolver.get_master()
            tf.Session.reset(tpu_grpc_url)
        else:
            tpu_cluster_resolver = None

        start_steps = []
        end_steps = []
        for lr in self.learning_rates:
            ext_dir = "{}_encoder".format(self.enc_type.name)
            if self.video_emb_type:
                ext_dir = os.path.join(ext_dir, "{}_video_emb".format(self.video_emb_type.name))
            lr_model_dir = os.path.join(self.model_dir, ext_dir, 'lr_{}'.format(lr))
            log.log_utils.add_lr_logging(os.path.join(lr_model_dir, 'logs'))

            lr_logger = logging.getLogger('main.vid2sentence.lr')
            lr_logger.info("Starting Model: {} \n"
                           "with learning rate {}".format(self, lr))

            run_config = tf.contrib.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                model_dir=lr_model_dir,
                log_step_count_steps=hp.TPU_LOG_STEP,
                session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
                tpu_config=tf.contrib.tpu.TPUConfig(tpu_iterations, tpu_shards),
                save_checkpoints_steps=hp.SAVE_CHK_STEP,
                keep_checkpoint_max=hp.KEEP_CHK
            )

            # set the classifier
            estimator = tf.contrib.tpu.TPUEstimator(
                model_fn=self.model_fn,
                model_dir=lr_model_dir,  # needed?
                config=run_config,
                params={'dropout_rate': self.dropout_rate,
                        'num_heads': self.mult_heads,
                        'feed_forward_hidden_units': self.ff_hidden_units,
                        'normalization_fn': self.normalization_fn,
                        'activation': self.activation,
                        'video_emb_type': self.video_emb_type,
                        'layer_stack': self.layer_stack,
                        'num_vocab': len(self.dataset.w2v_model.vocabulary),
                        'learning_rate': lr,
                        'bfloat16_mode': self.bfloat16_mode,
                        'optimizer_type': self.optimizer_type,
                        'adam_beta1': self.adam_beta1,
                        'adam_beta2': self.adam_beta2,
                        'adam_epsilon': self.adam_epsilon,
                        'adafactor_decay_rate': self.adafactor_decay_rate,
                        'adafactor_beta1': self.adafactor_beta1,
                        'adafactor_epsilon1': self.adafactor_epsilon1,
                        'adafactor_epsilon2': self.adafactor_epsilon2,
                        'label_smoothing': self.label_smoothing,
                        'weight_mode': self.weight_mode,
                        'input_data_type': self.dataset.input_data_type(),
                        'enc_type': self.enc_type,
                        'tpu': True,
                        'emb_lookup_table': self.dataset.w2v_model.emb_lookup_table,  # TODO delete test
                        'use_tpu': use_tpu,
                        },
                use_tpu=use_tpu,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                warm_start_from=ws)
            try:
                start_steps.append(estimator.get_variable_value("global_step"))
            except ValueError:
                start_steps.append(0)
            try:
                best_loss = float("inf")
                best_step = 0
                cur_epoch = 0
                hooks = list()
                hooks.append(tf.train.ProfilerHook(1, output_dir=lr_model_dir))

                while True:
                    cur_epoch += 1
                    if epochs and epochs < cur_epoch:
                        break
                    assert train_examples_per_epoch % train_batch_size == 0, \
                        "train_examples ({})_per_epoch must be divisible by train_batch_size ({})".format(
                            train_examples_per_epoch, train_batch_size)
                    assert eval_examples_per_epoch % eval_batch_size == 0, \
                        "eval_examples ({})_per_epoch must be divisible by eval_batch_size ({})".format(
                            eval_examples_per_epoch, eval_batch_size)
                    estimator.train(input_fn=train_input_fn, steps=int(train_examples_per_epoch / train_batch_size),
                                    hooks=hooks)
                    eval_dict = estimator.evaluate(input_fn=eval_input_fn,
                                                   steps=int(eval_examples_per_epoch / eval_batch_size))

                    if eval_dict['global_step'] >= self.min_step:
                        if eval_dict['loss'] < best_loss - self.delta_value:
                            best_loss = eval_dict['loss']
                            best_step = eval_dict['global_step']
                        if best_step < eval_dict['global_step'] - self.delta_step:
                            lr_logger.info(
                                "Lr: {}. Early stopping bc best loss {} and best step {} < {}. cur loss = {}".format(
                                    lr,
                                    best_loss,
                                    best_step,
                                    eval_dict['global_step'],
                                    eval_dict['loss']))
                            break

            except tf.train.NanLossDuringTrainingError:
                lr_logger.warning("NaN error with lr {}".format(lr))
            end_steps.append(estimator.get_variable_value("global_step"))
        return start_steps, end_steps, self.learning_rates

    def predict(self, input_files, beam_size, model_dir, with_labels=False, transfer_ckpt=None, array_input=False,
                tpu_iterations=hp.TPU_ITERATIONS,
                tpu_shards=hp.TPU_NUM_SHARDS, predict_batch_size=hp.TPU_PREDICT_BATCH_SIZE):

        self.logger.info("Start prediction with: Vid2Sentence Model Object: {}".format(self))
        if not array_input:
            dataset = input_files

            def input_fn(params):
                return data.creation.vid2sentence.tpu_input_fn(tfrecord_filepath=dataset.predict_data_path,
                                                               embedding_size=dataset.w2v_model.embedding_size,
                                                               sentence_max=dataset.sentence_max,
                                                               frames_max=dataset.frames_max,
                                                               is_training=False,
                                                               frame_shape=self.frame_shape,
                                                               input_data_type=dataset.input_data_type(),
                                                               with_labels=with_labels,
                                                               bfloat16_mode=self.bfloat16_mode,
                                                               params=params)
        else:
            input_fn, batch_size = \
                data.creation.vid2sentence.get_pred_input_fn_and_bsize(input_files,
                                                                       self.dataset.frames_path,
                                                                       self.dataset.input_data_type(),
                                                                       self.dataset.frames_max,
                                                                       self.dataset.max_height,
                                                                       self.dataset.max_width,
                                                                       transfer_ckpt)

        ws = None
        vars_to_start = None
        if self.video_emb_type == const.VideoEmbeddingType.InceptionV4:
            vars_to_start = '.*InceptionV4.*'
        elif self.video_emb_type == const.VideoEmbeddingType.MobilenetV1:
            vars_to_start = '.*MobilenetV1.*'
        if self.warm_start_dir:
            ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=self.warm_start_dir,
                                                vars_to_warm_start=vars_to_start)

        tpu_cluster_resolver = None

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=model_dir,
            log_step_count_steps=hp.TPU_LOG_STEP,
            session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(tpu_iterations, tpu_shards),
            save_checkpoints_steps=hp.SAVE_CHK_STEP,
            keep_checkpoint_max=hp.KEEP_CHK
        )

        # set the classifier
        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=self.model_fn,
            model_dir=model_dir,  # needed?
            config=run_config,
            params={'dropout_rate': self.dropout_rate,
                    'num_heads': self.mult_heads,
                    'feed_forward_hidden_units': self.ff_hidden_units,
                    'normalization_fn': self.normalization_fn,
                    'activation': self.activation,
                    'video_emb_type': self.video_emb_type,
                    'layer_stack': self.layer_stack,
                    'num_vocab': len(self.dataset.w2v_model.vocabulary),
                    'bfloat16_mode': self.bfloat16_mode,
                    'input_data_type': self.dataset.input_data_type(),
                    'enc_type': self.enc_type,
                    'tpu': True,
                    'pred_batch_size': 1,  # TODO
                    'pred_label_emb_size': self.dataset.w2v_model.embedding_size,
                    'pred_max_sentence': self.dataset.sentence_max,
                    'emb_lookup_table': self.dataset.w2v_model.emb_lookup_table,  # TODO delete test
                    'with_labels': with_labels,
                    'use_tpu': False,
                    'word2id': self.dataset.w2v_model.word2id,
                    'beam_size': beam_size,
                    'beam_alpha': hp.ALPHA,
                    },
            use_tpu=False,
            predict_batch_size=predict_batch_size,
            warm_start_from=ws, )

        id2word = self.dataset.w2v_model.id2word
        predictions = estimator.predict(input_fn)

        pred_sentences = []
        pred_probs = []
        pred_probs_softmax = []
        pred_video_ids = []
        for pred in predictions:
            pred_labels = pred['labels']
            words = []
            for labels in pred_labels:
                words.append([id2word[w_id] for w_id in labels])
            pred_sentences.append(words)
            pred_probs.append(pred['probs'])
            pred_probs_softmax.append(pred['probs_softmax'])
            pred_video_ids.append(pred['video_ids'])

        return pred_sentences, pred_probs, pred_probs_softmax, pred_video_ids


def mem_vid2sentence():
    w2v_model = models.words2vec.Words2vec('something_something_with_start_token', const.SOMETHING_LABELS_FILENAME,
                                           embedding_size=128, neg_samples=64)
    use_tpu = True
    model_dir = os.path.join(const.BASE_PATH, "testtest")

    if use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            hp.TPU_NAME,
            zone=hp.TPU_ZONE,
            project=hp.TPU_GCP_NAME)
        tpu_grpc_url = tpu_cluster_resolver.get_master()
        tf.Session.reset(tpu_grpc_url)
    else:
        tpu_cluster_resolver = None

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=model_dir,
        log_step_count_steps=hp.TPU_LOG_STEP,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(hp.TPU_ITERATIONS, hp.TPU_NUM_SHARDS),
        save_checkpoints_steps=hp.SAVE_CHK_STEP,
        keep_checkpoint_max=hp.KEEP_CHK
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=models.vid2sentence_modules.vid2sentence,
        model_dir=model_dir,  # needed?
        config=run_config,
        params={'dropout_rate': hp.V2S_DROPOUT_RATE,
                'num_heads': hp.V2S_NUM_MULT_HEADS,
                'feed_forward_hidden_units': hp.V2S_FF_HIDDEN_UNITS,
                'normalization_fn': models.vid2sentence_utils.layer_normalization,
                'activation': hp.V2S_ACTIVATION,
                'video_emb_type': const.VideoEmbeddingType.MobilenetV1,
                'layer_stack': hp.V2S_LAYERS,
                'num_vocab': len(w2v_model.vocabulary),
                'learning_rate': 0.02,
                'bfloat16_mode': True,
                'optimizer_type': hp.V2S_OPTIMIZER_TYPE,
                'adam_beta1': hp.V2S_ADAM_BETA1,
                'adam_beta2': hp.V2S_ADAM_BETA2,
                'adam_epsilon': hp.V2S_ADAM_EPSILON,
                'adafactor_decay_rate': hp.V2S_ADAFACTOR_DECAY,
                'adafactor_beta1': hp.V2S_ADAFACTOR_BETA,
                'adafactor_epsilon1': hp.V2S_ADAFACTOR_EPSILON1,
                'adafactor_epsilon2': hp.V2S_ADAFACTOR_EPSILON2,
                'label_smoothing': hp.V2S_LBL_SMOOTHING,
                'weight_mode': hp.V2S_WEIGHT_MODE,
                'input_data_type': const.InputDataTypes.OpticalFlowVideoJpg,
                'enc_type': const.EncoderType.MultiHeadAttention,
                'tpu': True,
                'emb_lookup_table': w2v_model.emb_lookup_table,  # TODO delete test
                'use_tpu': use_tpu,
                },
        use_tpu=use_tpu,
        train_batch_size=128,
        eval_batch_size=128)

    hooks = [] # [tf.train.ProfilerHook(1, output_dir=model_dir)]
    # builder = tf.profiler.ProfileOptionBuilder
    # opts = builder(builder.time_and_memory()).order_by('micros').build()
    # opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    # with tf.contrib.tfprof.ProfileContext(os.path.join(model_dir, 'profiler'), trace_steps=range(10, 20),
    #                                       dump_steps=[1]) as pctx:
    #     # Run online profiling with 'op' view and 'opts' options at step 15, 18, 20.
    #     pctx.add_auto_profiling('graph', opts, [0])
    #     # Run online profiling with 'scope' view and 'opts2' options at step 20.
    #     pctx.add_auto_profiling('scope', opts2, [0])
    estimator.train(input_fn=data.creation.vid2sentence.test_input_fn, hooks=hooks, steps=2)

if __name__ == '__main__':
    mem_vid2sentence()