# -*- coding: utf-8 -*-
import os
import numpy as np

import tensorflow as tf

from data_load import get_batch
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, load_hparams
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.logdir)

def beam_search(x, sess, g, batch_size=hp.batch_size):
    ''' it's not used in current version, will be added in the future'''
    inputs = np.reshape(np.transpose(np.array([x] * hp.beam_size), (1, 0, 2)),
                        (hp.beam_size * batch_size, hp.max_len))
    preds = np.zeros((batch_size, hp.beam_size, hp.y_max_len), np.int32)
    prob_product = np.zeros((batch_size, hp.beam_size))
    stc_length = np.ones((batch_size, hp.beam_size))

    for j in range(hp.y_max_len):
        _probs, _preds = sess.run(
            g.preds, {g.x: inputs, g.y: np.reshape(preds, (hp.beam_size * batch_size, hp.y_max_len))})
        j_probs = np.reshape(_probs[:, j, :], (batch_size, hp.beam_size, hp.beam_size))
        j_preds = np.reshape(_preds[:, j, :], (batch_size, hp.beam_size, hp.beam_size))
        if j == 0:
            preds[:, :, j] = j_preds[:, 0, :]
            prob_product += np.log(j_probs[:, 0, :])
        else:
            add_or_not = np.asarray(np.logical_or.reduce([j_preds > hp.end_id]), dtype=np.int)
            tmp_stc_length = np.expand_dims(stc_length, axis=-1) + add_or_not
            tmp_stc_length = np.reshape(tmp_stc_length, (batch_size, hp.beam_size * hp.beam_size))

            this_probs = np.expand_dims(prob_product, axis=-1) + np.log(j_probs) * add_or_not
            this_probs = np.reshape(this_probs, (batch_size, hp.beam_size * hp.beam_size))
            selected = np.argsort(this_probs / tmp_stc_length, axis=1)[:, -hp.beam_size:]

            tmp_preds = np.concatenate([np.expand_dims(preds, axis=2)] * hp.beam_size, axis=2)
            tmp_preds[:, :, :, j] = j_preds[:, :, :]
            tmp_preds = np.reshape(tmp_preds, (batch_size, hp.beam_size * hp.beam_size, hp.y_max_len))

            for batch_idx in range(batch_size):
                prob_product[batch_idx] = this_probs[batch_idx, selected[batch_idx]]
                preds[batch_idx] = tmp_preds[batch_idx, selected[batch_idx]]
                stc_length[batch_idx] = tmp_stc_length[batch_idx, selected[batch_idx]]

    final_selected = np.argmax(prob_product / stc_length, axis=1)
    final_preds = []
    for batch_idx in range(batch_size):
        final_preds.append(preds[batch_idx, final_selected[batch_idx]])

    return final_preds

logging.info("# Prepare test batches")
test_batches, num_test_batches, num_test_samples  = get_batch(hp.test1, hp.test1,
                                              1000, 1000,
                                              hp.vocab, hp.test_paraphrased, hp.test_batch_size,
                                              shuffle=False, paraphrase_type=hp.paraphrase_type)
iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
xs, ys, x_paraphrased_dict, _  = iter.get_next()

test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
m = Transformer(hp)
y_hat, _ = m.eval(xs, ys, x_paraphrased_dict)

logging.info("# Session")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.logdir)
    ckpt = hp.logdir if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)

    sess.run(test_init_op)

    logging.info("# get hypotheses")
    hypotheses = get_hypotheses(num_test_batches, num_test_samples, sess, y_hat, m.idx2token)

    logging.info("# write results")
    model_output = ckpt.split("/")[-1]
    if not os.path.exists(hp.testdir): os.makedirs(hp.testdir)
    translation = os.path.join(hp.testdir, model_output)
    with open(translation, 'w') as fout:
        fout.write("\n".join(hypotheses))