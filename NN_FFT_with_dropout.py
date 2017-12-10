# Kenneth Hall
# Learn to perform Fourier Transformation with Neural Networks

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os


curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, 'data')

input_train_path = os.path.join(data_dir, 'noisydata_train.csv')
real_train_path = os.path.join(data_dir, 'noisydata_fft_real_train.csv')
imag_train_path = os.path.join(data_dir, 'noisydata_fft_imag_train.csv')

input_test_path = os.path.join(data_dir, 'noisydata_test.csv')
real_test_path = os.path.join(data_dir, 'noisydata_fft_real_test.csv')
imag_test_path = os.path.join(data_dir, 'noisydata_fft_imag_test.csv')


FLAGS = None
sample_length = 200

disp_analysis = True

# Parameters
learning_rate = .0001
train_steps = 80000
dropout_ratio = 0.5
batch_size = 500

def fournn(x):
    """fournn builds the graph to perform FFT with a neural network

    Args:
        x: input tensor with dimensions (num_samples, sample_length).

    Returns:
        A tensor of shape (num_samples, sample_length) where each row is
        the approximated real-values from the FFT of the corresponding
        input row.
    """

    # First fully connected layer
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([200, 400])
        b_fc1 = bias_variable([400])

        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # Second fully connected layer
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([400,300])
        b_fc2 = bias_variable([300])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # Dropout - minimizes overfitting
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # Third fully connected layer
    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([300,200])
        b_fc3 = bias_variable([200])

        h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    return h_fc3, keep_prob


# Define how we create weight variables and bias variables
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Create plots to show learning progress
def plot_progress(fft_truth, fft_predicted, error_plot, n):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(fft_truth)
    plt.title('True FFT')
    plt.subplot(212)
    plt.plot(fft_predicted,'r')
    plt.title('Predicted FFT')
    plt.figure(2)
    plt.plot(error_plot,'g')
    plt.title('Squared Difference Error')
    plt.show()


def build_basic_graph():
    # Input tensor
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 200])

    # Build the FFN
    y_pred, keep_prob = fournn(x)

    return x, y_pred, keep_prob


def build_training_graph(learn, pred):
    # True y tensor (targets, or labels)
    with tf.name_scope('correct'):
        y_true = tf.placeholder(tf.float32, [None, 200])

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.pow(y_true - pred, 2))

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learn).minimize(loss)
    return y_true, loss, train_step


# generators will loop forever if batch_size > samples, also it has the chance to miss a
# few samples each iteration, though they all have equal probability, so it shouldnt matter
def data_generator(data, size):
    np.random.shuffle(data)
    sample_length = data.shape[0]
    curr = sample_length
    loop = 0
    while True:
        if curr+size > sample_length:
            curr = 0
            np.random.shuffle(data)
            loop += 1
            print('looping training data for the {0} time'.format(loop))
            continue
        x = data[curr:curr+size, 0]
        y_real = data[curr:curr+size, 1]
        y_imag = data[curr:curr+size, 2]
        curr += size
        yield x, y_real, y_imag


def get_data(size):
    # Import data and goal output data
    ## Training
    signal_data = np.loadtxt(open(input_train_path, 'r'), delimiter=',')
    fft_real_data = np.loadtxt(open(real_train_path, 'r'), delimiter=',')
    fft_imag_data = np.loadtxt(open(imag_train_path, 'r'), delimiter=',')

    # combining all data in order to make it easier to shuffle
    all_train = np.empty(shape=[3, signal_data.shape[0], signal_data.shape[1]],
        dtype=signal_data.dtype)
    all_train[0] = signal_data
    all_train[1] = fft_real_data
    all_train[2] = fft_imag_data
    all_train = all_train.swapaxes(0, 1)
    print('Number of training examples', all_train.shape[0])

    ## Testing
    signal_data_test = np.loadtxt(open(input_test_path, 'r'), delimiter=',')
    fft_real_data_test = np.loadtxt(open(real_test_path, 'r'), delimiter=',')
    fft_imag_data_test = np.loadtxt(open(imag_test_path, 'r'), delimiter=',')

    # combining all data in order to make it easier to shuffle
    all_test = np.empty(shape=[3, signal_data_test.shape[0], signal_data_test.shape[1]],
        dtype=signal_data_test.dtype)
    all_test[0] = signal_data_test
    all_test[1] = fft_real_data_test
    all_test[2] = fft_imag_data_test
    all_test = all_test.swapaxes(0, 1)
    print('Number of test examples', all_test.shape[0])

    assert(all_train.shape[0] >= size)
    assert(all_test.shape[0] >= size)
    return data_generator(all_train, size), data_generator(all_test, size)

# Create plots to show learning progress
def plot_training_progress(train_loss, test_loss):
    plt.figure(1)
    plt.plot(train_loss,'b')
    plt.plot(test_loss,'r')
    plt.title('Training and Testing Loss vs. Training Step')
    plt.show()


def main():
    # data_imports
    train_gen, test_gen = get_data(batch_size)

    # build graph
    x, y_pred, keep_prob = build_basic_graph()
    y_true, loss, train_step = build_training_graph(learning_rate, y_pred)

    train_loss = []
    test_loss = []
    # training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_steps):
            signal_data, fft_real_data, fft_imag_data = next(train_gen)
            train_step.run(feed_dict = {x: signal_data, y_true: fft_real_data,
                keep_prob: dropout_ratio})

            loss_step = sess.run(loss, feed_dict={x: signal_data, y_true: fft_real_data,
                keep_prob: 1.0})

            print('Step %i: Loss: %f' % (i, loss_step))
            train_loss.append(loss_step)
            signal_data_test, fft_real_data_test, fft_imag_data_test = next(test_gen)
            l = sess.run(loss, feed_dict={x: signal_data_test, y_true: fft_real_data_test, keep_prob: 1.0})
            test_loss.append(l)

            # disp_analysis = False
            if disp_analysis and i % 1000 == 0:
                # Run some analysis and display True FFT vs. Predicted FFT
                original_fft_data_showme = np.float32(fft_real_data)
                y_showme_pred = np.float32(sess.run(y_pred, feed_dict={x: signal_data,
                    y_true: fft_real_data, keep_prob: 1.0}))

                graph_choice = random.randint(0, len(signal_data)-1)
                error_plot = np.square(np.array(original_fft_data_showme[graph_choice]) -
                    np.array(y_showme_pred[graph_choice]))

                plot_progress(original_fft_data_showme[graph_choice],
                    y_showme_pred[graph_choice], error_plot, 1)


                # Test step
                signal_data_test, fft_real_data_test, fft_imag_data_test = next(test_gen)
                l = sess.run(loss, feed_dict={x: signal_data_test,
                    y_true: fft_real_data_test, keep_prob: 1.0})

                print('Testing loss at step %i: %f' % (i, l))
                y_showme_pred_test = np.float32(sess.run(y_pred,
                    feed_dict={x: signal_data_test, y_true: fft_real_data_test, keep_prob: 1.0}))

                graph_choice_test = random.randint(0, len(signal_data_test)-1)
                error_plot_test = np.square(np.array(fft_real_data_test[graph_choice_test]) -
                	np.array(y_showme_pred_test[graph_choice_test]))

                plot_progress(fft_real_data_test[graph_choice_test],
                    y_showme_pred_test[graph_choice_test], error_plot_test, 2)

                plot_training_progress(train_loss, test_loss)

if __name__ == "__main__":
    main()
