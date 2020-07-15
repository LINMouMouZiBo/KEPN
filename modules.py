# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from utils import create_pretrained_emb_from_txt

def ln(inputs, epsilon = 1e-8, scope="ln"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def get_token_embeddings(vocab_size, num_units, token2idx, embedding_file, zero_pad=True):
    with tf.variable_scope("shared_weight_matrix"):
        if embedding_file is None or not tf.gfile.Exists(embedding_file):
            print("# embedding is init randomly.")
            embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
        else:
            print("# embedding is init from file {}.".format(embedding_file))
            embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   initializer=create_pretrained_emb_from_txt(token2idx,embedding_file,vocab_size))
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings

def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
        outputs /= d_k ** 0.5
        outputs = mask(outputs, Q, K, type="key")

        if causality:
            outputs = mask(outputs, type="future")

        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        outputs = mask(outputs, Q, K, type="query")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs

def mask(inputs, queries=None, keys=None, type=None):
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        masks = tf.expand_dims(masks, 1) # (N, 1, T_k)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
    elif type in ("q", "query", "queries"):
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

        outputs = inputs*masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs

def multihead_attention(queries, keys, values,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries, d_model, use_bias=False, kernel_initializer=tf.initializers.random_normal(
          stddev=0.02, seed=None))
        K = tf.layers.dense(keys, d_model, use_bias=False, kernel_initializer=tf.initializers.random_normal(
          stddev=0.02, seed=None))
        V = tf.layers.dense(values, d_model, use_bias=False, kernel_initializer=tf.initializers.random_normal(
          stddev=0.02, seed=None))
        
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
              
        outputs += queries
        outputs = ln(outputs)
 
    return outputs

def ff(inputs, num_units, scope="positionwise_feedforward"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal(
          stddev=0.02, seed=None))

        outputs = tf.layers.dense(outputs, num_units[1], kernel_initializer=tf.initializers.random_normal(
          stddev=0.01, seed=None))

        outputs += inputs
        outputs = ln(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)
    
def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen+2)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def paraphrased_positional_encoding(inputs,
                                    maxlen,
                                    dims,
                                    masking=True,
                                    scope="positional_encoding"):
    E = dims # static
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        position_ind = inputs
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen+2)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        return tf.to_float(outputs)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)