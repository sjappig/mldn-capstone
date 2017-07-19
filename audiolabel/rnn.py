import collections

import numpy as np
import tensorflow as tf

import audiolabel.util


_Graph = collections.namedtuple('_Graph', ('x_in', 'nonpadded_lengths_in', 'logits_out', 'y_out', 'is_training'))

_Trainable = collections.namedtuple('_Optimizer', ('y_in', 'loss_out', 'optimization_operation_out'))


def _summary(name, tensor):
    tf.summary.scalar('{}_min'.format(name), tf.reduce_min(tensor))
    tf.summary.scalar('{}_max'.format(name), tf.reduce_max(tensor))
    tf.summary.scalar('{}_mean'.format(name), tf.reduce_mean(tensor))


def _rnn_layers(x, nonpadded_lengths, shape):
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.LSTMCell(
            shape[1], initializer=tf.orthogonal_initializer(gain=0.1)
        )
        for _ in range(0, shape[0])
    ])

    rnn_outputs, _ =  tf.nn.dynamic_rnn(
        cell,
        x,
        sequence_length=nonpadded_lengths,
        dtype=tf.float32
    )

    for idx in range(0, len(cell.weights)):
        _summary('rnn_weights_{}'.format(idx), cell.weights[idx])

    # Pick the last output without padded values
    output_indices = tf.stack([
        tf.range(tf.shape(x)[0]),
        nonpadded_lengths-1
    ], axis=1)

    output = tf.gather_nd(rnn_outputs, output_indices)

    return output


def _output_layer(x, num_outputs, is_training):
    W_output = tf.get_variable(
        "W_output",
        shape=[x.shape[1].value, num_outputs],
        initializer=tf.orthogonal_initializer(gain=0.1),
    )
#    W_output = tf.Variable(tf.truncated_normal([
#        x.shape[1].value,
#        num_outputs,
#    ], stddev=0.01))

    b_output = tf.Variable(tf.zeros(num_outputs))

    _summary('W_output', W_output)
    _summary('b_output', b_output)

    activations = tf.add(tf.matmul(x, W_output), b_output)

    return tf.contrib.layers.batch_norm(activations, is_training=is_training)


def _graph(num_classes, num_features, padded_length, rnn_shape):
    x = tf.placeholder(tf.float32, [None, padded_length, num_features], name='x')

    is_training = tf.placeholder(tf.bool, name='is_training')

    nonpadded_lengths = tf.placeholder(tf.int32, [None], name='nonpadded_lengths')

    rnn_output = _rnn_layers(x, nonpadded_lengths, shape=rnn_shape)

    logits = _output_layer(rnn_output, num_classes, is_training)

    _summary('logits', logits)

    predictor = tf.round(tf.nn.sigmoid(logits))

    return _Graph(x, nonpadded_lengths, logits, predictor, is_training)


def _trainable(graph, **optimizer_args):
    y = tf.placeholder(tf.float32, graph.y_out.shape, name='y')

    loss = tf.losses.sigmoid_cross_entropy(
        y,
        label_smoothing=0.2,
        logits=graph.logits_out,
    )

    tf.summary.scalar('loss', loss)

    optimization_operation = tf.train.AdamOptimizer(**optimizer_args).minimize(loss)

    return _Trainable(y, loss, optimization_operation)


def train_graph(x_train, y_train, train_lengths, x_validation, y_validation, val_lengths, batch_size=256, num_epochs=2000):
    num_classes = len(y_train[0])
    num_features = len(x_train[0][0])
    padded_length = len(x_train[0])

    print '{} classes, {} features, padded length is {}'.format(
        num_classes,
        num_features,
        padded_length,
    )
    graph = _graph(
        num_classes=num_classes,
        num_features=num_features,
        padded_length=padded_length,
        rnn_shape=(1, 64),
    )
    trainable = _trainable(graph, learning_rate=0.001)

    with tf.Session() as sess:
        batch_idx = 0
        writer = tf.summary.FileWriter('logs', graph=sess.graph)
        merged = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())

        for epoch in range(0, num_epochs):
            batches = audiolabel.util.batches(x_train, y_train, train_lengths, batch_size=batch_size)

            losses = []

            for x_batch, y_batch, len_batch in batches:
                loss, _, summary = sess.run([
                    trainable.loss_out, trainable.optimization_operation_out, merged
                ], feed_dict={
                    graph.x_in: x_batch,
                    graph.nonpadded_lengths_in: len_batch,
                    graph.is_training: True,
                    trainable.y_in: y_batch,
                })
                writer.add_summary(summary, batch_idx)
                batch_idx = batch_idx + 1
                losses.append(loss)

            y_pred, logits = sess.run([graph.y_out, graph.logits_out], feed_dict={
                graph.x_in: x_train,
                graph.nonpadded_lengths_in: train_lengths,
                graph.is_training: False,
            })

            f1_train = audiolabel.util.f1_score(y_train, y_pred)

            y_pred, logits = sess.run([graph.y_out, graph.logits_out], feed_dict={
                graph.x_in: x_validation,
                graph.nonpadded_lengths_in: val_lengths,
                graph.is_training: False,
            })

            f1_validation = audiolabel.util.f1_score(y_validation, y_pred)
            print 'Epoch {}: F1-score {}, {}; loss: {}'.format(epoch, f1_train, f1_validation, np.sum(losses))
#            import pdb; pdb.set_trace()

