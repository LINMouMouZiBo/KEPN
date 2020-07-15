# -*- coding: utf-8 -*-
import tensorflow as tf
from utils import calc_num_batches

def load_vocab(vocab_fpath):
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r', encoding="utf8").read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token

def load_data(fpath1, fpath2, paraphrased_fpath, maxlen1, maxlen2):
    sents1, sents2, paraphrased_pairs = [], [], []
    with open(fpath1, 'r', encoding="utf8") as f1, open(fpath2, 'r', encoding="utf8") as f2,\
            open(paraphrased_fpath, 'r', encoding="utf8") as f3:
        for sent1, sent2, dict_pair in zip(f1, f2, f3):
            if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            sents1.append(sent1.strip())
            sents2.append(sent2.strip())
            paraphrased_pairs.append(dict_pair.strip())
    return sents1, sents2, paraphrased_pairs


def generator_fn(sents1, sents2, paraphrased_pairs, vocab_fpath, paraphrase_type=0):
    token2idx, _ = load_vocab(vocab_fpath)
    for sent1, sent2, parap_pair in zip(sents1, sents2, paraphrased_pairs):
        sent1, sent2 = sent1.decode("utf-8"), sent2.decode("utf-8")
        in_words = sent1.split() + ["</s>"]
        x = [token2idx.get(t, token2idx["<unk>"]) for t in in_words]
        y = [token2idx.get(t, token2idx["<unk>"]) for t in ["<s>"] + sent2.split() + ["</s>"]]
        decoder_input, y = y[:-1], y[1:]
        x_paraphrase_dict = []
        word_set,pos_set = set(),set()
        for t in parap_pair.decode("utf-8").split():
            tem1, tem2 = t.split("->")
            if paraphrase_type == 0:
                word_set.add(tem1)
                x_paraphrase_dict.append([token2idx.get(tem1, token2idx["<unk>"]), token2idx.get(tem2, token2idx["<unk>"])])
            else:
                pos_set.add(int(tem2 if tem2!="<unk>" else 0))
                x_paraphrase_dict.append([token2idx.get(tem1, token2idx["<unk>"]), int(tem2 if tem2!="<unk>" else 0)])
        synonym_label = [i in word_set if paraphrase_type else w in word_set for i, w in enumerate(in_words)]

        x_seqlen, y_seqlen, xp_seqlen = len(x), len(y), len(x_paraphrase_dict)
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2), \
              (x_paraphrase_dict, xp_seqlen, len(x)), (synonym_label, x_seqlen, x)

def input_fn(sents1, sents2, paraphrased_pairs, vocab_fpath, batch_size, shuffle=False, paraphrase_type=0):
    shapes = (([None], (), ()),
              ([None], [None], (), ()),
              ([None, 2], (), ()),
              ([None], (), [None]),)
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.int32),
             (tf.int32, tf.int32, tf.int32))
    paddings = ((0, 0, ''),
                (0, 0, 0, ''),
                (0, 0, 0),
                (0, 0, 0))

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, paraphrased_pairs, vocab_fpath, paraphrase_type))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, paraphrased_fpath, batch_size, shuffle=False, paraphrase_type=0):
    sents1, sents2, paraphrased_pairs = load_data(fpath1, fpath2, paraphrased_fpath, maxlen1, maxlen2)
    batches = input_fn(sents1, sents2, paraphrased_pairs, vocab_fpath, batch_size, shuffle=shuffle, paraphrase_type=paraphrase_type)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)
