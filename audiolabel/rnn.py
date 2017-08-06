import collections

import numpy as np
import tensorflow as tf

import audiolabel.util


_Graph = collections.namedtuple('_Graph', (
    'x_in',
    'nonpadded_lengths_in',
    'logits_out',
    'y_out',
    'is_training_in',
))

_Trainable = collections.namedtuple('_Optimizer', (
    'loss_out',
    'optimization_operation_out',
))


def _weight_initializer():
    return tf.orthogonal_initializer()


def _rnn_layer(x, nonpadded_lengths, num_units):
    cell = tf.nn.rnn_cell.LSTMCell(
        num_units,
        initializer=_weight_initializer(),
    )

    rnn_outputs, _ =  tf.nn.dynamic_rnn(
        cell,
        x,
        sequence_length=nonpadded_lengths,
        dtype=tf.float32
    )

    # Pick the last output without padded values
    output_indices = tf.stack([
        tf.range(tf.shape(x)[0]),
        nonpadded_lengths-1
    ], axis=1)

    output = tf.gather_nd(rnn_outputs, output_indices, name='ignore_padded')

    return output


def _output_layer(x, num_outputs, is_training):
    W_output = tf.get_variable(
        "W",
        shape=[x.shape[1].value, num_outputs],
        initializer=_weight_initializer(),
    )

    b_output = tf.get_variable(name='b', shape=[num_outputs], initializer=tf.zeros_initializer())

    logits = tf.add(tf.matmul(x, W_output), b_output, name='logits')

    normalized = tf.contrib.layers.batch_norm(
        logits,
        is_training=is_training,
        fused=True,
        center=True,
    )

    return normalized

def _graph(x_batch, len_batch, num_classes, num_features, padded_length, rnn_num_units):
    print 'rnn_num_units: {}'.format(rnn_num_units)

    with tf.name_scope('test_input'):
        x_in = tf.placeholder_with_default(
            np.zeros([1, padded_length, num_features], dtype=np.float32),
            [None, padded_length, num_features],
            name='x_in',
        )
        nonpadded_lengths_in = tf.placeholder_with_default(
            np.zeros([1], dtype=np.int32),
            [None],
            name='nonpadded_lengths_in',
        )
    is_training_in = tf.placeholder(tf.bool, name='is_training_in')

    x, nonpadded_lengths = tf.cond(
        is_training_in,
        lambda: (x_batch, len_batch),
        lambda: (x_in, nonpadded_lengths_in),
        name='input_selector',
    )

    with tf.name_scope('rnn_layer'):
        rnn_output = _rnn_layer(x, nonpadded_lengths, rnn_num_units)

    with tf.name_scope('output_layer'):
        logits = _output_layer(rnn_output, num_classes, is_training_in)

    with tf.name_scope('prediction'):
        predictor = tf.round(tf.nn.sigmoid(logits), name='y_pred')

    return _Graph(x_in, nonpadded_lengths_in, logits, predictor, is_training_in)


def _get_weights(y):
    # These weights are calculated from the whole training set.
    # Each weight is inversely proportional to corresponding class frequency.
    class_weights = tf.constant([
        1., 8.54679803, 1.1034088, 3.55824446, 1.04745231, 2.66513057, 3.76519097
    ], name='class_weights')
    weights = tf.multiply(y, class_weights)
    weights = tf.reduce_mean(weights, axis=1)

    return tf.reshape(weights, [-1, 1])


def _trainable(graph, y_batch, **optimizer_args):
    weights = _get_weights(y_batch)

    loss = tf.losses.sigmoid_cross_entropy(
        y_batch,
        weights=weights,
        logits=graph.logits_out,
    )

    optimization_operation = tf.train.AdamOptimizer(
        **optimizer_args
    ).minimize(loss)

    return _Trainable(loss, optimization_operation)


def _predict(sess, graph, x, lengths):
    y_batches = []

    for x_batch, len_batch in audiolabel.util.batches(x, lengths, batch_size=2048):
        y_batch = sess.run(graph.y_out, feed_dict={
            graph.x_in: x_batch,
            graph.nonpadded_lengths_in: len_batch,
            graph.is_training_in: False,
        })

        y_batches.append(y_batch)

    return np.concatenate(y_batches)


def _train_graph(x_train, y_train, train_lengths, num_epochs, batch_size=256, x_validation=None, y_validation=None, validation_lengths=None):
    num_classes = len(y_train[0])
    num_features = len(x_train[0][0])
    padded_length = len(x_train[0])

    batch_size = min(len(x_train), batch_size)

    print '{} classes, {} features, padded length is {}'.format(
        num_classes,
        num_features,
        padded_length,
    )

    x_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    len_tensor = tf.convert_to_tensor(train_lengths, dtype=tf.int32)

    num_batches = int(np.ceil(len(x_train) / float(batch_size)))

    x_batch, len_batch, y_batch = tf.train.batch(
        [x_tensor, len_tensor, y_tensor],
        batch_size=batch_size,
        enqueue_many=True,
        name='train_input',
    )

    graph = _graph(
        x_batch=x_batch,
        len_batch=len_batch,
        num_classes=num_classes,
        num_features=num_features,
        padded_length=padded_length,
        rnn_num_units=512,
    )

    with tf.name_scope('training'):
        trainable = _trainable(
            graph,
            y_batch=y_batch,
            epsilon=1e-4,
        )

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(0, num_epochs):
            losses = []

            for batch in range(0, num_batches):
                loss, _ = sess.run([
                    trainable.loss_out, trainable.optimization_operation_out,
                ], feed_dict={
                    graph.is_training_in: True,
                })
                losses.append(loss)

            if (epoch % 50 == 0) or (epoch + 1 == num_epochs):
                y_pred = _predict(sess, graph, x_train, train_lengths)
                f1_train = audiolabel.util.f1_score(y_train, y_pred, average='weighted')

                if x_validation is not None:
                    y_pred = _predict(sess, graph, x_validation, validation_lengths)
                    f1_validation = audiolabel.util.f1_score(y_validation, y_pred, average='weighted')

                else:
                    f1_validation = 'N/A'

                print 'Epoch {}: F1-score train: {}; validation: {}; loss: {}'.format(
                    epoch,
                    f1_train,
                    f1_validation,
                    np.sum(losses),
                )

        coord.request_stop()
        coord.join(threads)

        saver = tf.train.Saver()
        saver.save(sess, 'RNN.ckpt')

    def wrapped_predict(x, nonpadded_lengths):
        with tf.Session() as sess:
            saver.restore(sess, 'RNN.ckpt')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            y_pred = _predict(sess, graph, x, nonpadded_lengths)

            coord.request_stop()
            coord.join(threads)

        return y_pred

    return audiolabel.util.Predictor(wrapped_predict, use_only_x=False)


def create(x_train, y_train, **kwargs):
    return _train_graph(x_train, y_train, **kwargs)
