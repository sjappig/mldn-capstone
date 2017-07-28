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


def _summary(name, tensor):
    tf.summary.scalar('{}_min'.format(name), tf.reduce_min(tensor))
    tf.summary.scalar('{}_max'.format(name), tf.reduce_max(tensor))
    tf.summary.scalar('{}_mean'.format(name), tf.reduce_mean(tensor))


def _weight_initializer():
    return tf.orthogonal_initializer()


def _rnn_layers(x, nonpadded_lengths, num_units):
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

#    for idx in range(0, len(cell.weights)):
#        _summary('rnn_weights_{}'.format(idx), cell.weights[idx])

    # Pick the last output without padded values
    output_indices = tf.stack([
        tf.range(tf.shape(x)[0]),
        nonpadded_lengths-1
    ], axis=1)

    output = tf.gather_nd(rnn_outputs, output_indices)

    return output


def _dropout_layer(x):
    return tf.layers.dropout(inputs=x, rate=0.25)

def _output_layer(x, num_outputs, is_training):
    W_output = tf.get_variable(
        "W_output",
        shape=[x.shape[1].value, num_outputs],
        initializer=_weight_initializer(),
    )

    b_output = tf.Variable(tf.zeros(num_outputs))

    activations = tf.add(tf.matmul(x, W_output), b_output)

    return tf.contrib.layers.batch_norm(activations, is_training=is_training, fused=True, center=True)


def _graph(x_batch, len_batch, num_classes, num_features, padded_length, rnn_num_units):
    print 'rnn_num_units: {}'.format(rnn_num_units)

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
    )

    rnn_output = _rnn_layers(x, nonpadded_lengths, rnn_num_units)

#    dropout_output = _dropout_layer(rnn_output)

    logits = _output_layer(rnn_output, num_classes, is_training_in)

    predictor = tf.round(tf.nn.sigmoid(logits))

    return _Graph(x_in, nonpadded_lengths_in, logits, predictor, is_training_in)


def _get_weights(y):
    # These weights are calculated from the whole training set.
    # Each weight is inversely proportional to corresponding class frequency.
    class_weights = tf.constant([
        1., 8.54679803, 1.1034088, 3.55824446, 1.04745231, 2.66513057, 3.76519097
    ])
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

#    tf.summary.scalar('loss', loss)

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


def train_graph(x_train, y_train, train_lengths, x_validation, y_validation, validation_lengths, batch_size=256, num_epochs=3000):
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
    )

    graph = _graph(
        x_batch=x_batch,
        len_batch=len_batch,
        num_classes=num_classes,
        num_features=num_features,
        padded_length=padded_length,
        rnn_num_units=512,
    )
    trainable = _trainable(
        graph,
        y_batch=y_batch,
#        learning_rate=0.001, #1e-4,
#        epsilon=1e-4,
    )

    with tf.Session() as sess:
        batch_idx = 0
#        writer = tf.summary.FileWriter('logs', graph=sess.graph)
#        merged = tf.summary.merge_all()

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
#                writer.add_summary(summary, batch_idx)
                batch_idx = batch_idx + 1
                losses.append(loss)

            if (epoch % 50 == 0)  or (epoch + 1 == num_epochs):
                y_pred = _predict(sess, graph, x_train, train_lengths)
                f1_train = audiolabel.util.f1_score(y_train, y_pred, average='weighted')

                y_pred = _predict(sess, graph, x_validation, validation_lengths)
                f1_validation = audiolabel.util.f1_score(y_validation, y_pred, average='weighted')

                print 'Epoch {}: F1-score train: {}; validation: {}; loss: {}'.format(
                    epoch,
                    f1_train,
                    f1_validation,
                    np.sum(losses),
                )

        coord.request_stop()
        coord.join(threads)
