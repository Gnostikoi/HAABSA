#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from sklearn.metrics import f1_score
import sys
import os

import os


FLAGS = tf.compat.v1.app.flags.FLAGS
#general variables
tf.compat.v1.app.flags.DEFINE_integer("year",2016, "year data set [2014]")
# tf.compat.v1.app.flags.DEFINE_integer("year",2015, "year data set [2014]")
tf.compat.v1.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 20, 'number of example per batch')
tf.compat.v1.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.compat.v1.app.flags.DEFINE_float('learning_rate', 0.09, 'learning rate')
tf.compat.v1.app.flags.DEFINE_integer('n_class', 13, 'number of distinct class')
# tf.compat.v1.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.compat.v1.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.compat.v1.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
tf.compat.v1.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.compat.v1.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.compat.v1.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.compat.v1.app.flags.DEFINE_integer('n_iter', 100, 'number of train iter')
tf.compat.v1.app.flags.DEFINE_float('keep_prob1', 0.5, 'dropout keep prob')
tf.compat.v1.app.flags.DEFINE_float('keep_prob2', 0.5, 'dropout keep prob')
tf.compat.v1.app.flags.DEFINE_float('lambda_0', 0.3, 'portion of sentence loss')
tf.compat.v1.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.compat.v1.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.compat.v1.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
tf.compat.v1.app.flags.DEFINE_string('is_r', '1', 'prob')
tf.compat.v1.app.flags.DEFINE_integer('max_target_len', 19, 'max target length')
tf.compat.v1.app.flags.DEFINE_float('threshold', 0.75, 'probability larger than this value will be regarded as true.')

# traindata, testdata adn embeddings
tf.compat.v1.app.flags.DEFINE_string("train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'traindata'+str(FLAGS.year)+".txt", "train data path")
tf.compat.v1.app.flags.DEFINE_string("test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'testdata'+str(FLAGS.year)+".txt", "formatted test data path")
tf.compat.v1.app.flags.DEFINE_string("embedding_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'embedding'+str(FLAGS.year)+".txt", "pre-trained glove vectors file path")
tf.compat.v1.app.flags.DEFINE_string("remaining_test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

#svm traindata, svm testdata
tf.compat.v1.app.flags.DEFINE_string("train_svm_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'trainsvmdata'+str(FLAGS.year)+".txt", "train data path")
tf.compat.v1.app.flags.DEFINE_string("test_svm_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'testsvmdata'+str(FLAGS.year)+".txt", "formatted test data path")
tf.compat.v1.app.flags.DEFINE_string("remaining_svm_test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingsvmtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

#hyper traindata, hyper testdata
tf.compat.v1.app.flags.DEFINE_string("hyper_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertraindata'+str(FLAGS.year)+".txt", "hyper train data path")
tf.compat.v1.app.flags.DEFINE_string("hyper_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevaldata'+str(FLAGS.year)+".txt", "hyper eval data path")

tf.compat.v1.app.flags.DEFINE_string("hyper_svm_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertrainsvmdata'+str(FLAGS.year)+".txt", "hyper train svm data path")
tf.compat.v1.app.flags.DEFINE_string("hyper_svm_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevalsvmdata'+str(FLAGS.year)+".txt", "hyper eval svm data path")

#external data sources
tf.compat.v1.app.flags.DEFINE_string("pretrain_file", "data/externalData/glove.42B."+str(FLAGS.embedding_dim)+"d.txt", "pre-trained glove vectors file path")
tf.compat.v1.app.flags.DEFINE_string("train_data", "data/externalData/restaurant_train_"+str(FLAGS.year)+".xml",
                    "train data path")
tf.compat.v1.app.flags.DEFINE_string("test_data", "data/externalData/restaurant_test_"+str(FLAGS.year)+".xml",
                    "test data path")
tf.compat.v1.app.flags.DEFINE_string("summary_path", 'summary/', "summary path")

tf.compat.v1.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.compat.v1.app.flags.DEFINE_string('results/prob_file', 'prob1.txt', 'prob')
tf.compat.v1.app.flags.DEFINE_string('saver_file', 'prob1.txt', 'prob')


def print_config():
    #FLAGS._parse_flags()
    FLAGS(sys.argv)
    print('\nParameters:')
    for k, v in sorted(tf.compat.v1.app.flags.FLAGS.flag_values_dict().items()):
        print('{}={}'.format(k, v))


def loss_func(y, prob):
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = - tf.reduce_mean(y* tf.log(prob)) + sum(reg_loss)
    return loss

def acc_func(y, prob, y_sen, prob_sen, thre=0.5):
    y = tf.cast(tf.argmax(y,1), tf.int32)
    prob = tf.cast(tf.argmax(prob, 1), tf.int32)
    y_sen = tf.cast(y_sen, tf.float64)
    prob_sen = tf.cast(tf.math.greater_equal(prob_sen, [thre]), tf.float64)

    correct_pred = tf.equal(prob,y)
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    f1s = [0, 0, 0]
    # y_true = tf.cast(multi_y, tf.float64)
    # y_pred = tf.cast(multi_pred, tf.float64)
    y_true = y_sen
    y_pred = prob_sen

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s

    return acc_num, acc_prob, micro, macro, weighted


def train_func(loss, r, global_step, optimizer=None):
    if optimizer:
        return optimizer(learning_rate=r).minimize(loss, global_step=global_step)
    else:
        return tf.train.AdamOptimizer(learning_rate=r).minimize(loss, global_step=global_step)


def summary_func(loss, acc, f1, test_loss, test_acc, test_f1, _dir, title, sess):
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc = tf.summary.scalar('acc' + title, acc)
    summary_f1_micro = tf.summary.scalar('f1_micro' + title, f1)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)
    test_summary_f1_micro = tf.summary.scalar('f1_micro' + title, test_f1)
    train_summary_op = tf.summary.merge([summary_loss, summary_acc, summary_f1_micro])
    validate_summary_op = tf.summary.merge([summary_loss, summary_acc, summary_f1_micro])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc, test_summary_f1_micro])
    train_summary_writer = tf.summary.FileWriter(FLAGS.summary_path + "train/" + _dir, sess.graph)
    test_summary_writer = tf.summary.FileWriter(FLAGS.summary_path + "test/" + _dir, sess.graph)
    validate_summary_writer = tf.summary.FileWriter(FLAGS.summary_path + "validate/" + _dir, sess.graph)
    return train_summary_op, test_summary_op, validate_summary_op, \
        train_summary_writer, test_summary_writer, validate_summary_writer


def saver_func(_dir):
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)
    import os
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return saver






