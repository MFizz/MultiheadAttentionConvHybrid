import yaml
import logging.config
import logging
import coloredlogs
import constants.constants as const
import tensorflow as tf


def add_main_logging(main_log_path=const.MAIN_LOGS,
                  default_main_conf_file=const.DEFAULT_MAIN_CONF_FILE,
                  default_level=logging.INFO):
    _setup_logging(main_log_path, default_main_conf_file, default_level)


def add_words2vec_logging(model_log_path,
                      default_model_conf_file=const.DEFAULT_WORDS2VEC_CONF_FILE,
                      default_level=logging.INFO):
    _setup_logging(model_log_path, default_model_conf_file, default_level)


def add_vid2sentence_logging(model_log_path,
                      default_model_conf_file=const.DEFAULT_VID2SENTENCE_CONF_FILE,
                      default_level=logging.INFO):
    _setup_logging(model_log_path, default_model_conf_file, default_level)


def add_lr_logging(model_log_path,
                      default_model_conf_file=const.DEFAULT_LR_CONF_FILE,
                      default_level=logging.INFO):
    _setup_logging(model_log_path, default_model_conf_file, default_level)


def _setup_logging(log_path,
                  default_conf_file,
                  default_level=logging.INFO):

    # create log directories
    # tensorflow bug in MakeDirs revert when fixed TODO
    tf.gfile.MakeDirs(log_path)
    if not const.GCLOUD and tf.gfile.Exists(default_conf_file):
        with tf.gfile.GFile(default_conf_file, mode='r') as f:
            try:
                config = yaml.safe_load(f.read())
                _set_handler_path(config, log_path)
                logging.config.dictConfig(config)
                coloredlogs.install()
            except Exception as e:
                print(e)
                print('Error in Logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print('Failed to load configuration file. Using default configs')


def _set_handler_path(config, path=None):
    handlers = dict(config['handlers'])
    for handler in handlers:
        if handler == 'console':
            continue
        config['handlers'][handler]['filename'] = config['handlers'][handler]['filename'].format(path)

