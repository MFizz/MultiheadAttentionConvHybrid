import tensorflow as tf
import constants.constants as const
import models.vid2sentence_utils

class LossTest(tf.test.TestCase):

    def test_get_default_weights(self):
        expected_weights = tf.convert_to_tensor([[1., 1., 1.],
                                                 [1., 1., 1.]])
        logits = tf.ones((2, 3, 5))
        weights = models.vid2sentence_utils.get_loss_weights(logits)
        self.assertAllClose(weights, expected_weights)
        self.assertAllClose(tf.reduce_sum(weights), 6.)

    def test_get_linear_loss_weights(self):
        expected_weights = tf.convert_to_tensor([[1.5, 1., 0.5],
                                                 [1.5, 1., 0.5]])
        logits = tf.ones((2, 3, 5))
        weights = models.vid2sentence_utils.get_loss_weights(logits, weight_mode=const.WeightMode.Linear)
        self.assertAllClose(weights, expected_weights)
        self.assertAllClose(tf.reduce_sum(weights), 6.)

        expected_weights = tf.convert_to_tensor([[1.904762, 1.809524, 1.714286, 1.619048, 1.52381 , 1.428571, 1.333333,
                                                  1.238095, 1.142857, 1.047619, 0.952381, 0.857143, 0.761905, 0.666667,
                                                  0.571429, 0.47619, 0.380952, 0.285714, 0.1904762, 0.0952381],
                                                 [1.904762, 1.809524, 1.714286, 1.619048, 1.52381, 1.428571, 1.333333,
                                                  1.238095, 1.142857, 1.047619, 0.952381, 0.857143, 0.761905, 0.666667,
                                                  0.571429, 0.47619, 0.380952, 0.285714, 0.1904762, 0.0952381]])
        logits = tf.ones((2, 20, 5))
        weights = models.vid2sentence_utils.get_loss_weights(logits, weight_mode=const.WeightMode.Linear)
        self.assertAllClose(weights, expected_weights)
        self.assertAllClose(tf.reduce_sum(weights), 40.)

    def test_get_gauss_loss_weights(self):
        expected_weights = tf.convert_to_tensor([[1.343974, 0.981855, 0.674171],
                                                 [1.343974, 0.981855, 0.674171]])
        logits = tf.ones((2, 3, 5))
        weights = models.vid2sentence_utils.get_loss_weights(logits, weight_mode=const.WeightMode.Gauss)
        self.assertAllClose(weights, expected_weights)
        self.assertAllClose(tf.reduce_sum(weights), 6.)

        expected_weights = tf.convert_to_tensor([[2.022023, 1.47721, 1.014297, 0.918584, 0.910721, 0.91048, 0.910478,
                                                  0.910478, 0.910478, 0.910478, 0.910478, 0.910478, 0.910478, 0.910478,
                                                  0.910478, 0.910478, 0.910478, 0.910478, 0.9104775, 0.9104775],
                                                 [2.022023, 1.47721, 1.014297, 0.918584, 0.910721, 0.91048, 0.910478,
                                                  0.910478, 0.910478, 0.910478, 0.910478, 0.910478, 0.910478, 0.910478,
                                                  0.910478, 0.910478, 0.910478, 0.910478, 0.9104775, 0.9104775]])
        logits = tf.ones((2, 20, 5))
        weights = models.vid2sentence_utils.get_loss_weights(logits, weight_mode=const.WeightMode.Gauss)
        self.assertAllClose(weights, expected_weights)
        self.assertAllClose(tf.reduce_sum(weights), 40.)

    def test_loss(self):
        one_hot_labels = tf.convert_to_tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                               [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                                               [[0, 1, 0], [0, 0, 1], [1, 0, 0]]])

        logits1 = tf.convert_to_tensor([[[100, 0, 0], [0, 100, 0], [0, 0, 100]],
                                       [[0, 0, 100], [100, 0, 0], [0, 100, 0]],
                                       [[0, 100, 0], [0, 0, 100], [100, 0, 0]]], dtype=tf.float32)

        logits2 = tf.convert_to_tensor([[[0, 100, 0], [0, 0, 100], [0, 100, 0]],
                                        [[0, 100, 0], [0, 100, 0], [100, 0, 0]],
                                        [[0, 0, 100], [100, 0, 0], [0, 100, 0]]], dtype=tf.float32)

        logits3 = tf.convert_to_tensor([[[0, 100, 0], [0, 100, 0], [0, 100, 0]],
                                        [[0, 0, 100], [0, 100, 0], [100, 0, 0]],
                                        [[0, 0, 100], [100, 0, 0], [100, 0, 0]]], dtype=tf.float32)

        weights1 = tf.convert_to_tensor([[1, 1, 1],
                                         [1, 1, 1],
                                         [1, 1, 1]])

        weights2 = tf.convert_to_tensor([[0, 3, 0],
                                         [3, 0, 0],
                                         [0, 0, 3]])

        weights3 = tf.convert_to_tensor([[1.5, 1, 0.5],
                                         [1.5, 1, 0.5],
                                         [1.5, 1, 0.5]])

        loss1 = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits1)
        self.assertAllClose(loss1, 0.)

        # loss = 6.3 for 20 wordcount
        loss1_1 = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits1, label_smoothing=0.01)
        self.assertAllClose(loss1_1, 0.666667)

        loss2 = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits2)
        self.assertAllClose(loss2, 100.)

        loss3 = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits3)
        self.assertAllClose(loss3, 66.666664)

        loss3_1 = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits3, weights=weights1)
        self.assertAllClose(loss3_1, 66.666664)

        loss3_2 = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits3, weights=weights2)
        self.assertAllClose(loss3_2, 0.)

        loss3_3 = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits3, weights=weights3)
        self.assertAllClose(loss3_3, 66.666664)


if __name__ == '__main__':
    tf.test.main()
