import tensorflow as tf
import data
import constants.constants as const
import models

if __name__ == '__main__':
    proj_name = 'something_something_jpg'
    w2v_model = models.words2vec.Words2vec('something_something_eva_test', const.SOMETHING_LABELS_FILENAME,
                                           embedding_size=128, neg_samples=64)
    dataset = data.creation.vid2sentence.V2SDataset(proj_name, w2v_model,
                                                             const.SOMETHING_VIDEO_BASE_PATH,
                                                             prediction_labels_filename=const.SOMETHING_EVAL_LABELS_FILENAME,
                                                             prediction_name='test')
    dataset = data.creation.vid2sentence.input_fn(tfrecord_filepath=dataset.predict_data_path,
                                        batch_size=2,  # TODO
                                        embedding_size=dataset.w2v_model.embedding_size,
                                        sentence_max=dataset.sentence_max,
                                        frames_max=dataset.frames_max,
                                        is_training=False,
                                        frame_shape=None,
                                        input_data_type=dataset.input_data_type(),
                                        with_labels=True)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:

        for i in range(100):
            value = sess.run(next_element)
            print(value)