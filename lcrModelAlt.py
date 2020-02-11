#!/usr/bin/env python
# encoding: utf-8

import os, sys
import pickle
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from nn_layer import dynamic_rnn, softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from att_layer import dot_produce_attention_layer, bilinear_attention_layer, mlp_attention_layer, Mlp_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_twitter
import numpy as np
from operator import itemgetter



def lcr_rot(n_asp, input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, l2, _id='all'):
    print('I am lcr_rot_alt.')
    cell = tf.contrib.rnn.LSTMCell
    # left hidden
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
    pool_l = reduce_mean_with_len(hiddens_l, sen_len_fw)

    # right hidden
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')
    pool_r = reduce_mean_with_len(hiddens_r, sen_len_bw)

    # target hidden
    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)


    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl')
    outputs_t_l = tf.squeeze(tf.matmul(att_l, hiddens_l))
    # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr')
    outputs_t_r = tf.squeeze(tf.matmul(att_r, hiddens_r))

    # attention target left
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l')
    outputs_l = tf.squeeze(tf.matmul(att_t_l, hiddens_t))
    # attention target right
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r')
    outputs_r = tf.squeeze(tf.matmul(att_t_r, hiddens_t))

    for i in range(2):
        # attention target
        att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl'+str(i))
        outputs_t_l = tf.squeeze(tf.matmul(att_l, hiddens_l))

        att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr'+str(i))
        outputs_t_r = tf.squeeze(tf.matmul(att_r, hiddens_r))

        # attention left
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l'+str(i))
        outputs_l = tf.squeeze(tf.matmul(att_t_l, hiddens_t))
        # attention right
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r'+str(i))
        outputs_r = tf.squeeze(tf.matmul(att_t_r, hiddens_t))

    outputs = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    prob = softmax_layer(outputs, 8 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, l2, FLAGS.n_class)
    
    asp_size = tf.size(n_asp)
    begin = tf.map_fn(lambda x: tf.math.reduce_sum(n_asp[:x],0), tf.range(asp_size), tf.int32)
    prob_sen = tf.map_fn( 
                lambda i: tf.clip_by_value(
                    tf.reduce_sum(tf.slice(prob, [begin[i], 0], [n_asp[i], FLAGS.n_class]), axis=0), \
                    clip_value_min = 0. , clip_value_max = 1.), \
                tf.range(asp_size), \
                dtype = tf.float32)

    return prob, prob_sen, outputs, att_l, att_r, att_t_l, att_t_r


def main(train_path, test_path, accuracyOnt, test_size, remaining_size, momentum=0.85):
    # print_config()
    l2 = FLAGS.l2_reg
    learning_rate = FLAGS.learning_rate

    with tf.device('/gpu:1'):
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        word_embedding = tf.constant(w2v, name='word_embedding')

        keep_prob1 = tf.constant(FLAGS.keep_prob1, tf.float32)
        keep_prob2 = tf.constant(FLAGS.keep_prob2,tf.float32)

        lambda_0 = tf.constant(FLAGS.lambda_0, tf.float32)
        lambda_1 = tf.constant(1 - FLAGS.lambda_0, tf.float32)

        with tf.name_scope('inputs'):
            y_sen = tf.placeholder(tf.float32, [None, FLAGS.n_class], name='y_sentence_level')
            n_asp = tf.placeholder(tf.int32, [None], name='n_asp')

            x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='x')
            y = tf.placeholder(tf.float32, [None, FLAGS.n_class], name='y')
            sen_len = tf.placeholder(tf.int32, None, name='sentence_length')

            x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='x_backwards')
            sen_len_bw = tf.placeholder(tf.int32, [None], name='sentence_length_backwards')

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len], name='target_words')
            tar_len = tf.placeholder(tf.int32, [None], name='target_length')

        inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
        inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
        target = tf.nn.embedding_lookup(word_embedding, target_words)

        alpha_fw, alpha_bw = None, None
        prob, prob_sen, _, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r = lcr_rot(n_asp, inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tar_len, keep_prob1, keep_prob2, l2, 'all')

        loss_asp = loss_func(y, prob)
        loss_sen = loss_func(y_sen, prob_sen)
        loss = lambda_1 * loss_asp + lambda_0 * loss_sen
        acc_num, acc_prob, f1_micro, f1_macro, f1_weighted = acc_func(y, prob, y_sen, prob_sen, thre = FLAGS.threshold)
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,  momentum=momentum).minimize(loss, global_step=global_step)
        # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
        # true_y = tf.argmax(y, 1)
        true_y = y_sen
        pred_y = tf.cast(tf.math.greater_equal(prob_sen, [FLAGS.threshold]), tf.int32)
        # pred_y = tf.argmax(prob, 1)

        title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
            FLAGS.keep_prob1,
            FLAGS.keep_prob2,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.l2_reg,
            FLAGS.max_sentence_len,
            FLAGS.embedding_dim,
            FLAGS.n_hidden,
            FLAGS.n_class
        )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import datetime
        # timestamp = str(int(time.time()))
        timestamp = datetime.datetime.now().isoformat()
        _dir = str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        test_f1_micro = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, f1_micro, test_loss, test_acc, test_f1_micro, _dir, title, sess)
        # validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        # saver = saver_func(save_dir)

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, '/-')

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_y_sen, tr_target_word, tr_tar_len, _, _, _ , tr_n_asp= load_inputs_twitter(
            train_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r, # reverse
            FLAGS.max_target_len
        )
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_y_sen, te_target_word, te_tar_len, _, _, _, te_n_asp = load_inputs_twitter(
            test_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, n_asp_b, yi, y_sen_i, target, tl, batch_size, is_shuffle=True):
            # for index in batch_index(len(yi), batch_size, 1, is_shuffle):
            for index in batch_index(len(n_asp_b), batch_size, 1, is_shuffle):
                selected_rows = itemgetter(*index)(list(n_asp_b.values()))
                r_index = []
                for idxs in selected_rows:
                    if idxs != []:
                        r_index.extend(idxs)
                _n_asp = np.asarray([len(tup) for tup in list(selected_rows) if len(tup) != 0])
                # print(f"length of _n_asp: {_n_asp.shape[0]}")
                feed_dict = {
                    x: x_f[r_index],
                    x_bw: x_b[r_index],
                    y: yi[r_index],
                    y_sen: y_sen_i[index],
                    n_asp: _n_asp,
                    sen_len: sen_len_f[r_index],
                    sen_len_bw: sen_len_b[r_index],
                    target_words: target[r_index],
                    tar_len: tl[r_index]
                }
                yield feed_dict, len(r_index)

        max_acc = 0.
        max_f1 = 0.
        max_fw, max_bw = None, None
        max_tl, max_tr = None, None
        max_ty, max_py = None, None
        max_prob = None
        step = None

        for i in range(FLAGS.n_iter):
            trainacc, trainf1, traincnt, train_batchcnt= 0., 0., 0, 0
            for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_n_asp, tr_y, \
                tr_y_sen, tr_target_word, tr_tar_len, FLAGS.batch_size):
                # _, step = sess.run([optimizer, global_step], feed_dict=train)
                _, step, summary, _trainacc, _trainf1 = sess.run([optimizer, global_step, train_summary_op, acc_num, f1_micro], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)
                trainacc += _trainacc            # saver.save(sess, save_dir, global_step=step)
                trainf1 += _trainf1
                traincnt += numtrain
                train_batchcnt += 1
            acc, f1, cost, cnt, test_batchcnt = 0., 0., 0., 0, 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_n_asp, te_y,
                                            te_y_sen, te_target_word, te_tar_len, 2000, False):
                if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                    _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                        [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob], feed_dict=test)
                    fw += list(_fw)
                    bw += list(_bw)
                    tl += list(_tl)
                    tr += list(_tr)
                else:
                    _loss, _acc, _f1, _ty, _py, _p, _fw, _bw, _tl, _tr = sess.run([loss, acc_num, f1_micro, true_y, pred_y, prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r], feed_dict=test)
                ty = np.asarray(_ty)
                py = np.asarray(_py)
                p = np.asarray(_p)
                fw = np.asarray(_fw)
                bw = np.asarray(_bw)
                tl = np.asarray(_tl)
                tr = np.asarray(_tr)
                acc += _acc
                f1 += _f1
                # cost += _loss * num
                cost += _loss
                cnt += num
                test_batchcnt += 1
            print('all samples={}, correct prediction={}'.format(cnt, acc))
            trainacc = trainacc / traincnt
            trainf1 = trainf1 / train_batchcnt
            acc = acc / cnt
            f1 = f1 / test_batchcnt
            totalacc = ((acc * remaining_size) + (accuracyOnt * (test_size - remaining_size))) / test_size
            # cost = cost / cnt
            cost = cost / test_batchcnt
            print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, train_f1_micro={:.6f}, test acc={:.6f}, \
                test_f1_micro={:.6f}, combined acc={:.6f}'.format(i, cost,trainacc, trainf1, acc, f1, totalacc))
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc, test_f1_micro: f1})
            test_summary_writer.add_summary(summary, step)
            # if acc > max_acc:
            if f1 > max_f1:
                max_acc = acc
                max_f1 = f1
                max_fw = fw
                max_bw = bw
                max_tl = tl
                max_tr = tr
                max_ty = ty
                max_py = py
                max_prob = p

        # encode training data
        train_feed_dict = {
                x: tr_x,
                x_bw: tr_x_bw,
                y: tr_y,
                y_sen: tr_y_sen,
                n_asp: tr_n_asp,
                sen_len: tr_sen_len,
                sen_len_bw: tr_sen_len_bw,
                target_words: tr_target_word,
                tar_len: tr_target_len
        }
        test_feed_dict = {
                x: te_x,
                x_bw: te_x_bw,
                y: te_y,
                y_sen: te_y_sen,
                n_asp: te_n_asp,
                sen_len: te_sen_len,
                sen_len_bw: te_sen_len_bw,
                target_words: te_target_word,
                tar_len: te_target_len
        }
        tr_outputs = sess.run([outputs], feed_dict = train_feed_dict)
        te_outputs = sess.run([outputs], feed_dict = test_feed_dict)

        with open("results/embeddings/train_emb.npy", 'wb') as f:
            np.save(f, tr_outputs)
        with open("results/embeddings/test_emb.npy", 'wb') as f:
            np.save(f, te_outputs)

        P = precision_score(max_ty, max_py, average='micro')
        R = recall_score(max_ty, max_py, average='micro')
        F1 = f1_score(max_ty, max_py, average='micro')
        print('(Individual aspect) P:', P, 'avg=', sum(P) / FLAGS.n_class)
        print('(Individual aspect) R:', R, 'avg=', sum(R) / FLAGS.n_class)
        print('(Individual aspect) F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)

        prob_data = {
            'forward_att': max_fw,
            'backward_att': max_bw,
            'target_left_att': max_tl,
            'target_right_att': max_tr,
            'true': max_ty,
            'predict': max_py
        }
        fp = open(FLAGS.prob_file, 'w')
        pickle.dump(prob_data, fp)
        # for item in max_prob:
        #     fp.write(' '.join([str(it) for it in item]) + '\n')
        # fp = open(FLAGS.prob_file + '_fw', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_fw):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_bw', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_bw):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_tl', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_tl):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_tr', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_tr):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

        print('Optimization Finished! Max acc={}, Max micro f1={}'.format(max_acc, max_f1))

        print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            FLAGS.learning_rate,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg
        ))

        return max_acc, np.where(np.subtract(max_py, max_ty) == 0, 0, 1), max_fw.tolist(), max_bw.tolist(), max_tl.tolist(), max_tr.tolist()


if __name__ == '__main__':
    tf.app.run()
