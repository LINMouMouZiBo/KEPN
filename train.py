# -*- coding: utf-8 -*-
import tensorflow as tf

from model import Transformer
from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs, get_hypotheses
import os
from hparams import Hparams
import math
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2,
                                             hp.maxlen1, hp.maxlen2,
                                             hp.vocab, hp.train_paraphrased, hp.batch_size,
                                             shuffle=True, paraphrase_type=hp.paraphrase_type)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2,
                                             1000, 1000,
                                             hp.vocab, hp.eval_paraphrased, hp.batch_size,
                                             shuffle=False, paraphrase_type=hp.paraphrase_type)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys, x_paraphrased_dict, synonym_label = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = Transformer(hp)
loss, train_op, global_step, train_summaries = m.train(xs, ys, x_paraphrased_dict, synonym_label)
y_hat, eval_summaries = m.eval(xs, ys, x_paraphrased_dict)
# y_hat = m.infer(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    from_scrtch = True
    if ckpt is None:
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    if from_scrtch:
        sess.run(global_step.assign(tf.zeros_like(global_step)))
    print("current is {} step".format(sess.run(global_step)))
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss) # train loss

            logging.info("#evaluation start")
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)

            model_output = "model_E%02dL%.2f" % (epoch, _loss)
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses))

            logging.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()


logging.info("Done")
