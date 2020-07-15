# -*- coding: utf-8 -*-
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from modules import paraphrased_positional_encoding
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, self.token2idx, self.hp.embedding_file, zero_pad=True)

    def encode(self, xs, training=True):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, sents1

    def decode(self, ys, x_paraphrased_dict, memory, training=True):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys
            x_paraphrased_dict, paraphrased_lens, paraphrased_sents = x_paraphrased_dict
            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            batch_size = tf.shape(decoder_inputs)[0] # (N, T2, 2)
            seqlens = tf.shape(decoder_inputs)[1]  # (N, T2, 2)
            paraphrased_lens = tf.shape(x_paraphrased_dict)[1]  # (N, T2, 2)

            x_paraphrased_o, x_paraphrased_p = x_paraphrased_dict[:,:,0], x_paraphrased_dict[:,:,1]

            x_paraphrased_o_embedding = tf.nn.embedding_lookup(self.embeddings, x_paraphrased_o)  # N, W2, d_model
            if self.hp.paraphrase_type == 0:
                x_paraphrased_p_embedding = tf.nn.embedding_lookup(self.embeddings, x_paraphrased_p)
            else:
                x_paraphrased_p_embedding = paraphrased_positional_encoding(x_paraphrased_p, self.hp.maxlen2, self.hp.d_model)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

            # add paraphrased dictionary attention
            h = tf.fill([batch_size, seqlens, paraphrased_lens, self.hp.d_model], 1.0) * tf.expand_dims(dec, axis=2)

            o_embeding = tf.fill([batch_size, seqlens, paraphrased_lens, self.hp.d_model], 1.0) * tf.expand_dims(x_paraphrased_o_embedding, axis=1)
            W_a_o = tf.get_variable("original_word_parameter_w", [2*self.hp.d_model], initializer=tf.initializers.random_normal(
          stddev=0.01, seed=None))
            V_a_o = tf.get_variable("original_word_parameter_v", [2*self.hp.d_model], initializer=tf.initializers.random_normal(
          stddev=0.01, seed=None))
            h_o_concat = tf.concat([h, o_embeding], -1) # N, T2, W2, 2*d_model
            score_tem_o = tf.tanh(W_a_o * h_o_concat) # N, T2, W2, 2*d_model
            score_o = tf.reduce_sum(V_a_o * score_tem_o, axis=-1) # N, T2, W2
            a = tf.nn.softmax(score_o) # N, T2, W2
            c_o = tf.matmul(a, x_paraphrased_o_embedding) # (N, T2, W2) * (N, W2, d_model) --> N, T2, d_model

            p_embeding = tf.fill([batch_size, seqlens, paraphrased_lens, self.hp.d_model], 1.0) * tf.expand_dims(x_paraphrased_p_embedding, axis=1)
            W_a_p = tf.get_variable("paraphrased_word_parameter_w", [2*self.hp.d_model], initializer=tf.initializers.random_normal(
          stddev=0.01, seed=None))
            V_a_p = tf.get_variable("paraphrased_word_parameter_v", [2*self.hp.d_model], initializer=tf.initializers.random_normal(
          stddev=0.01, seed=None))
            h_p_concat = tf.concat([h, p_embeding], -1) # N, T2, W2, 2*d_model
            score_tem_p = tf.tanh(W_a_p * h_p_concat) # N, T2, W2, 2*d_model
            score_p = tf.reduce_sum(V_a_p * score_tem_p, axis=-1) # N, T2, W2
            a = tf.nn.softmax(score_p) # N, T2, W2
            c_p = tf.matmul(a, x_paraphrased_p_embedding) # (N, T2, W2) * (N, W2, d_model) --> N, T2, d_model

            c_t = tf.concat([c_o, c_p], axis=-1) # N, T2, d_model --> N, T2, 2*d_model
            out_dec = tf.layers.dense(tf.concat([dec, c_t], axis=-1), self.hp.d_model, activation=tf.tanh, use_bias=False, kernel_initializer=tf.initializers.random_normal(
          stddev=0.01, seed=None))

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', out_dec, weights) # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2

    def labeling(self, x, menmory):
        synonym_label, seqlens, sents1 = x
        logits = tf.layers.dense(menmory, 2, activation=tf.tanh, use_bias=False,
                                  kernel_initializer=tf.initializers.random_normal(stddev=0.01, seed=None))
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        # Synonym Labeling loss
        y = tf.one_hot(synonym_label, depth=2)
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
        nonpadding = tf.to_float(tf.not_equal(sents1, self.token2idx["<pad>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
        return logits, y_hat, loss


    def train_labeling(self, xs, synonym_label=None):
        # forward
        memory, sents1 = self.encode(xs)
        _, _, loss = self.labeling(synonym_label, memory)

        # train scheme
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)
        summaries = tf.summary.merge_all()
        return loss, train_op, global_step, summaries

    def train(self, xs, ys, x_paraphrased_dict, synonym_label=None):
        # forward
        memory, sents1 = self.encode(xs)
        _, _, synonym_label_loss = self.labeling(synonym_label, memory)
        logits, preds, y, sents2 = self.decode(ys, x_paraphrased_dict, memory)

        # train scheme
        # generation loss
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
        # multi task loss
        tloss = self.hp.l_alpha * loss + (1.0-self.hp.l_alpha) * synonym_label_loss

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(tloss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("tloss", tloss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys, x_paraphrased_dict):
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1 = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, x_paraphrased_dict, memory, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        summaries = tf.summary.merge_all()

        return y_hat, summaries

