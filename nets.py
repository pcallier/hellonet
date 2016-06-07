#!/usr/bin/env python
"""

"""

import tensorflow as tf
import chainer

import data


def tf_lstm():
    vocab_size = len(data.default_vocab)
    lstm_hidden_size = 10
    sequence_width = 20
    batch_size = 100

    initializer = tf.random_uniform_initializer(-1,1)

    input_tensor = tf.placeholder(tf.float32, (lstm_hidden_size, batch_size, sequence_width))
    input_split = [tf.reshape(i, (batch_size, sequence_width)) for i in tf.split(0, lstm_hidden_size, input_tensor)]

    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size) 
    initial_state = lstm.zero_state(batch_size, tf.float32)
    outputs, states = tf.rnn(lstm, input_split, initial_state=initial_state)

    init_op = tf.initialize_all_variables()

    session = tf.Session()
    session.run(init_op)

    inputs = data.batch_inputs()
    feed = {input_tensor: np.random.rand(lstm_hidden_size, batch_size, sequence_width).astype('float32')}

    run_output = session.run(outputs, feed_dict=feed)

    print type(run_output)
    print len(run_output)

def tf_gru():
    vocab_size = len(data.default_vocab)
    hidden_size = 10
    batch_size = 100

    initializer = tf.random_uniform_initializer(-1,1)

    input_tensor = tf.placeholder(tf.float32, (lstm_hidden_size, batch_size, sequence_width))
    input_split = [tf.reshape(i, (batch_size, sequence_width)) for i in tf.split(0, lstm_hidden_size, input_tensor)]
    #  num_units = 8
    #  num_proj = 6
    #  state_size = num_units + num_proj
    #  batch_size = 3
    #  input_size = 2
    #  with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
    #    x = tf.zeros([batch_size, input_size])
    #    m = tf.zeros([batch_size, state_size])
    #    cell = tf.nn.rnn_cell.LSTMCell(
    #        num_units=num_units, num_proj=num_proj, forget_bias=1.0)
    #    output, state = cell(x, m)
    #    sess.run([tf.initialize_all_variables()])
    #    res = sess.run([output, state],
    #                   {x.name: np.array([[1., 1.], [2., 2.], [3., 3.]]),
    #                    m.name: 0.1 * np.ones((batch_size, state_size))})

    gru = tf.nn.rnn_cell.GruCell(lstm_hidden_size) 
    initial_state = lstm.zero_state(batch_size, tf.float32)
    outputs, states = tf.rnn(lstm, input_split, initial_state=initial_state)

    init_op = tf.initialize_all_variables()

    session = tf.Session()
    session.run(init_op)

    inputs = data.batch_inputs()
    feed = {input_tensor: np.random.rand(lstm_hidden_size, batch_size, sequence_width).astype('float32')}

    run_output = session.run(outputs, feed_dict=feed)

    print type(run_output)
    print len(run_output)

if __name__ == "__main__":
    tf_lstm()
    
