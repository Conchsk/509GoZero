# -*- coding: UTF-8 -*-
import random

import tensorflow as tf


class ResNet:
    def __init__(self, board_size):
        self.board_size = board_size

        self.boards = tf.placeholder(tf.float32, shape=[None, self.board_size, self.board_size, 17])
        self.policy = tf.placeholder(tf.float32, shape=[None, self.board_size * self.board_size])
        self.value = tf.placeholder(tf.float32, shape=[None, 1])

        first_layer = tf.layers.conv2d(self.boards, 256, 3, padding='same')
        first_layer = tf.layers.batch_normalization(first_layer)
        first_layer = tf.nn.relu(first_layer)

        res_net_tower = first_layer
        for i in range(10):
            res_net_tower = self._res_block(res_net_tower)

        self.policy_predict = self._policy_head(res_net_tower)
        self.value_predict = self._value_head(res_net_tower)

        self.loss = tf.reduce_mean(
            tf.square(self.value - self.value_predict) - self.policy * tf.log(self.policy_predict))
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _res_block(self, input_tensor):
        conv1 = tf.layers.conv2d(input_tensor, 256, 3, padding='same')
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        conv2 = tf.layers.conv2d(conv1, 256, 3, padding='same')
        conv2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.nn.relu(tf.add(input_tensor, conv2))
        return conv2

    def _policy_head(self, input_tensor):
        conv = tf.layers.conv2d(input_tensor, 2, 1, padding='same')
        conv = tf.layers.batch_normalization(conv)
        conv = tf.nn.relu(conv)
        conv_flat = tf.reshape(conv, shape=[-1, self.board_size * self.board_size * 2])
        policy_predict = tf.layers.dense(conv_flat, self.board_size * self.board_size + 1)
        return policy_predict

    def _value_head(self, input_tensor):
        conv = tf.layers.conv2d(input_tensor, 1, 1, padding='same')
        conv = tf.layers.batch_normalization(conv)
        conv = tf.nn.relu(conv)
        conv_flat = tf.reshape(conv, shape=[-1, self.board_size * self.board_size])
        fc1 = tf.layers.dense(conv_flat, 256)
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.layers.dense(fc1, 1)
        value_predict = tf.nn.tanh(fc2)
        return value_predict

    def train(self, boards_batch, policy_batch, value_batch):
        self.sess.run(self.train_step, feed_dict={
            self.boards: boards_batch,
            self.policy: policy_batch,
            self.value: value_batch
        })

    def predict(self, boards):
        policy_predict, value_predict = self.sess.run([self.policy_predict, self.value_predict],
                                                      feed_dict={self.boards: boards})
        return policy_predict, value_predict
