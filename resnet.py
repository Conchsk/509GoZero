import tensorflow as tf


class ResNet:
    def __init__(self):
        None

    def _res_block(self, input_tensor):
        l1_a = tf.layers.conv2d(inputs=input_tensor, filters=256, kernel_size=3)
        l1_b = tf.layers.batch_normalization(inputs=l1_a)
        l1_c = tf.nn.relu(features=l1_b)
        l2_a = tf.layers.conv2d(inputs=l1_c, filters=256, kernel_size=3)
        l2_b = tf.layers.batch_normalization(inputs=l2_a)
        l2_c = tf.nn.relu(features=tf.add(input_tensor, l2_b))
        return l2_c

    def _policy_head(self, input_tensor):
        l1_a = tf.layers.conv2d(inputs=input_tensor, filters=2, kernel_size=1)
        l1_b = tf.layers.batch_normalization(inputs=l1_a)
        l1_c = tf.nn.relu(features=l1_b)
        l2_a = tf.nn.softmax()

    def _value_head(self, input_tensor):
        l1_a = tf.layers.conv2d(inputs=input_tensor, filters=1, kernel_size=1)
        l1_b = tf.layers.batch_normalization(inputs=l1_a)
        l1_c = tf.nn.relu(features=l1_b)


    def predict(self, boards):


