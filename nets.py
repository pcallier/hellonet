#!/usr/bin/env python
"""

"""

import tensorflow as tf
import chainer

import data


def tf_lstm(input):
    vocab_size = len(data.default_vocab)
    lstm_size = 10
    lstm = tf.nn.rnn_cell.BasicLSTMCell(self, lstm_size,
                                        input_size=(lstm_size, vocab_size))
    