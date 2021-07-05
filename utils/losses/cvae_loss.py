import tensorflow as tf


def create_cvae_loss(model):

    def cvae_loss(y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(squared_difference, axis=-1)

    return cvae_loss
