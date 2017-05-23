#!/usr/bin/env python

'''
Machine learning for Iris dataset
  - Uses TensorFlow to create a Logistic Regression classifier
  - Full ML pipeline with tensorflow backend

Imports raw data from csv, randomizes data, preprocesses data,
splits data, trains model, tests model
'''
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

__version__ = '0.1.1'
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

def train(X_train, y_train, X_valid, y_valid, X_test, y_test, 
            learning_rate=0.05, load=False):
    # parameters for size of W, b, x, y
    n_samples, n_features = X_train.shape
    # comma here necessary to unpack tuple (?,)
    n_classes, = y_train[0].shape

    # variables for the weights and biases
    W = tf.Variable(tf.random_normal([n_features, n_classes], stddev=0.1), 
        name='weights')
    b = tf.Variable(tf.random_normal([n_classes], stddev=0.1),
        name='biases')

    # placeholders for the input -> the features and label
    x = tf.placeholder(tf.float32, [None, n_features], name='x')
    y_labels = tf.placeholder(tf.float32, [None, n_classes], name='y_labels')

    # necessary node -> variable initializer
    init = tf.global_variables_initializer()

    # node for softmax(X*W + b)
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # use tf cross entropy function and reduce mean
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y))

    # use gradient descent optimizer with given learning rate
    # the session will run objective -> minimize cross_entropy with g.d.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    objective = optimizer.minimize(cross_entropy)

    # determine if max value from softmax is equal to max value from label
    correct_prediction = tf.equal(tf.argmax(y, axis=1), 
            tf.argmax(y_labels, axis=1))
    # calculate the model accuracy on the set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # saver node to save/restore weights/biases
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # initialize the global vars
        sess.run(init)

        if not load:
            # train a new model
            for _ in range(1000):
                sess.run(objective, feed_dict={x: X_train, y_labels: y_train})

            # save the model
            save_path = saver.save(sess, 'saved_models/iris_model')
            print('Model saved in', save_path, end='\n\n')
        else:
            # import the model
            saver = tf.train.import_meta_graph('saved_models/iris_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('saved_models/'))
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

def main(load=False, visual=False):
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

    # run the training/testing
    train(X_train, y_train, X_valid, y_valid, X_test, y_test, load=load)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='iris.py',
        description='Train/test iris dataset using logistic regression')

    parser.add_argument('--load', dest='load', action='store_true',
        help='load model rather than train (default: False)')
    parser.add_argument('--visual', dest='visual', action='store_true',
        help='plot data and features prior to load/test (default: False)')

    args = parser.parse_args()

    main(load=args.load, visual=args.visual)
