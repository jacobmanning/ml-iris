#!/usr/bin/env python

'''
Machine learning for Iris dataset
  - Uses TensorFlow to create a Logistic Regression classifier
  - Full ML pipeline with TensorFlow backend

Imports raw data from csv, randomizes data, preprocesses data,
splits data, trains model, tests model, saves model

Uses TensorBoard to visualize the results
'''

import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

__version__ = '0.2.0'
__author__ = 'Jacob Manning'
__email__ = 'jacobmanning@pitt.edu'

def load_iris(filename):
    col_names = ['sepal_length', 'sepal_width','pedal_length',
                'pedal_width', 'label']

    # return dataframe labeled with col_names
    return pd.read_csv(filename, names=col_names)

def train_valid_test_split(df):
    # separate the data frame by label
    setosa = df[df.label == 'Iris-setosa']
    versicolor = df[df.label == 'Iris-versicolor']
    virginica = df[df.label == 'Iris-virginica']

    # split that dataframe into training, validation, and test
    s_tr, s_v, s_te = _split(setosa)
    ve_tr, ve_v, ve_te = _split(versicolor)
    vi_tr, vi_v, vi_te = _split(virginica)

    # concatenate the dataframes from each type into the threee sets
    train_set = pd.concat([s_tr, ve_tr, vi_tr], ignore_index=True)
    valid_set = pd.concat([s_v, ve_v, vi_v], ignore_index=True)
    test_set = pd.concat([s_te, ve_te, vi_te], ignore_index=True)

    return (train_set, valid_set, test_set)

def _split(df):
    # seed the random num generator for reproducable train, valid, test sets
    np.random.seed(12345)

    # randomize the data frame
    randomized_df = df.iloc[np.random.permutation(len(df))]
    
    # calculate indicies to split data by 60% train, 20% valid/test
    train_idx = int(len(randomized_df) * 0.6)
    offset = int(len(randomized_df) * 0.2)

    # split randomized data into the sets
    train_set = randomized_df.iloc[:train_idx]
    valid_set = randomized_df.iloc[train_idx:train_idx+offset]
    test_set = randomized_df.iloc[train_idx+offset:]

    return (train_set, valid_set, test_set)

def label_split(df):
    # get the labels for each example
    label = df['label']
    # one-hot-encode the labels into vectors
    one_hot_label = one_hot(label)

    # get all columns but the label for each example
    x = np.array(df.iloc[:, :-1], dtype=np.float32)

    return (x, one_hot_label)

def one_hot(arr):
    # representations of each class string with an integer
    class_translations = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

    # zero 2-d array to hold new one-hot labels
    encoded = np.zeros((len(arr), len(class_translations)))

    # set value corresponding to label as 1
    for i, label in enumerate(arr):
        encoded[i][class_translations[label]] = 1

    return encoded

def initial_plots(df):
    # plot all of the features in the dataframe against each other
    sns.pairplot(df, hue='label', size=2)
    plt.show()

def variable_summaries(var):
    # helper function to create summaries for tensorboard
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def train(X_train, y_train, X_valid, y_valid, X_test, y_test, 
            learning_rate=0.05, load=False, filename='iris_model', stddev=0.1):
    # parameters for size of W, b, x, y
    n_samples, n_features = X_train.shape
    # comma here necessary to unpack tuple (?,)
    n_classes, = y_train[0].shape

    # placeholders for the input -> the features and label
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_features], name='x')
        y_labels = tf.placeholder(tf.float32, [None, n_classes], name='y_labels')

    # variable for the weights
    with tf.name_scope('weights'):
        W = tf.Variable(tf.random_normal([n_features, n_classes], stddev=stddev), 
            name='weights')
        variable_summaries(W)

    # variable for the biases
    with tf.name_scope('biases'):
        b = tf.Variable(tf.random_normal([n_classes], stddev=stddev),
            name='biases')
        variable_summaries(b)

    # node for softmax(X*W + b)
    with tf.name_scope('softmax_Wx_plus_b'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        tf.summary.histogram('activations', y)

    # use tf cross entropy function and reduce mean
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        with tf.name_scope('optimizer'):
            # use gradient descent optimizer with given learning rate
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        with tf.name_scope('objective'):
            # the session will run objective -> minimize cross_entropy with g.d.
            objective = optimizer.minimize(cross_entropy)

    with tf.name_scope('evaluate'):
        with tf.name_scope('correct_prediction'):
            # determine if max softmax value is equal to max label value
            correct_prediction = tf.equal(tf.argmax(y, axis=1), 
                    tf.argmax(y_labels, axis=1))
        
        with tf.name_scope('accuracy'):
            # calculate the model accuracy on the set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('saver'):
        # saver node to save/restore weights/biases
        saver = tf.train.Saver()

    # merge all tf.summaries for tensorboard
    merged = tf.summary.merge_all()

    # necessary node -> variable initializer
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # initialize the global vars
        sess.run(init)

        if not load:
            # writers for tensorboard
            train_writer = tf.summary.FileWriter('tensorboard/' + 
                filename + '/train', sess.graph)
            test_writer = tf.summary.FileWriter('tensorboard/' + 
                filename + '/test')

            # train a new model
            for i in range(1000):
                if i % 50 == 0:
                    summary, acc = sess.run([merged, accuracy], 
                        feed_dict={x: X_valid, y_labels: y_valid})
                    test_writer.add_summary(summary, i)

                    print('Validation set accuracy at step {}: {}'.format(i, acc))
                else:
                    summary, _ = sess.run([merged, objective],
                        feed_dict={x: X_train, y_labels: y_train})
                    train_writer.add_summary(summary, i)

            print('\nView tensorboard with:')
            print('tensorboard --logdir=tensorboard/' + filename, end='\n\n')

            train_writer.close()
            test_writer.close()

            # save the model
            save_path = saver.save(sess, 'saved_models/' + filename)
            print('Model saved in', save_path, end='\n\n')
        else:
            # import the model
            saver = tf.train.import_meta_graph('saved_models/' + 
                filename + '.meta')
            saver.restore(sess, 'saved_models/' + filename)
            print('Model loaded successfully!', end='\n\n')

            # print the model parameters
            print('Weights')
            print(sess.run(W), end='\n\n')
            print('Biases')
            print(sess.run(b), end='\n\n')

            # test the model
            print('Validation accuracy:', end=' ')
            print(sess.run(accuracy, feed_dict={x: X_valid, y_labels: y_valid}))
            print('Test accuracy:', end=' ')
            print(sess.run(accuracy, feed_dict={x: X_test, y_labels: y_test}))

def main(load=False, visual=False, learning_rate=0.05, 
    filename='iris_model', stddev=0.1):
    # load and split the data
    df = load_iris('data/iris.csv')
    train_set, valid_set, test_set = train_valid_test_split(df)

    # visualize the data
    if visual:
        initial_plots(train_set)

    # split each set into the X (features) and y (labels)
    X_train, y_train = label_split(train_set)
    X_valid, y_valid = label_split(valid_set)
    X_test, y_test = label_split(test_set)

    # delete previous tensorboard files
    if not load:
        tb_dir = 'tensorboard/' + filename

        if tf.gfile.Exists(tb_dir):
            tf.gfile.DeleteRecursively(tb_dir)

    # run the training/testing
    train(X_train, y_train, X_valid, y_valid, X_test, y_test, load=load,
        learning_rate=learning_rate, filename=filename, stddev=stddev)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='iris.py',
        description='Train/test iris dataset using logistic regression')

    parser.add_argument('--load', dest='load', action='store_true',
        default=False, help='load model rather than train')
    parser.add_argument('--visual', dest='visual', action='store_true',
        default=False, help='plot data and features prior to load/test')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
        default=0.05, help='learning rate for GradientDescentOptimizer')
    parser.add_argument('--filename', dest='filename', type=str,
        default='iris_model', help='file to store/load model to/from')
    parser.add_argument('--stddev', dest='stddev', type=float,
        default=0.1, help='standard deviation for random_normal init values')

    args = parser.parse_args()

    main(load=args.load, visual=args.visual, learning_rate=args.learning_rate,
        filename=args.filename, stddev=args.stddev)
