# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
import os, re
import logging

logging.basicConfig(level=logging.INFO)

def calc_num_batches(total_num, batch_size):
    return total_num // batch_size + int(total_num % batch_size != 0)

def convert_idx_to_token_tensor(inputs, idx2token):
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)

def postprocess(hypotheses, idx2token):
    _hypotheses = []
    for h in hypotheses:
        sent = " ".join(idx2token[idx] for idx in h)
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("▁", " ") # remove bpe symbols
        _hypotheses.append(sent.strip())
    return _hypotheses

def save_hparams(hparams, path):
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)

def load_hparams(parser, path):
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        parser.f = v

def save_variable_specs(fpath):
    def _get_size(shp):
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")

def get_hypotheses(num_batches, num_samples, sess, tensor, dict):
    hypotheses = []
    for _ in tqdm(range(num_batches)):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses, dict)

    return hypotheses[:num_samples]

def load_embed_txt(embed_file):
  emb_dict = dict()
  emb_size = None
  with open(embed_file, "r", encoding="utf8") as f:
    for i, line in enumerate(f):
      tokens = line.strip().split("\t")
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      emb_dict[word] = vec
      if emb_size:
        if emb_size != len(vec):
          word = " "
          vec = list(map(float, tokens))
        assert emb_size == len(vec), "All embedding size should be same. %d, %d, %s" %(i, len(vec), word)
      else:
        emb_size = len(vec)
  return emb_dict, emb_size

def create_pretrained_emb_from_txt(token2idx, embed_file, vocab_size, dtype=tf.float32):
  vocab = list(token2idx.keys())[:vocab_size]
  emb_dict, emb_size = load_embed_txt(embed_file)

  for token in vocab:
    if token not in emb_dict:
      emb_dict[token] = np.random.standard_normal(emb_size)

  emb_mat = np.array([(emb_dict[token] if token in emb_dict else emb_dict["<unk>"]) for token in vocab],
      dtype=dtype.as_numpy_dtype())
  return emb_mat
