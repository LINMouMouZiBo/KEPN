import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # train
    ## files
    parser.add_argument('--train1', default='data/tcnp.train.src',
                             help="input training data")
    parser.add_argument('--train2', default='data/tcnp.train.tgt',
                             help="output training data")
    parser.add_argument('--eval1', default='data/tcnp.dev.src',
                             help="input evaluation data")
    parser.add_argument('--eval2', default='data/tcnp.dev.tgt',
                             help="output evaluation data")
    parser.add_argument('--paraphrase_type', default=1, type=int)
    parser.add_argument('--train_paraphrased', default='data/train_paraphrased_pair.txt',
                             help="train paraphrased pair dictionary")
    parser.add_argument('--eval_paraphrased', default='data/dev_paraphrased_pair.txt',
                             help="eval paraphrased pair dictionary")

    ## vocabulary
    parser.add_argument('--vocab', default='data/jieba.vocab', help="vocabulary file path")
    parser.add_argument('--embedding_file', default='', help="embedding file path")

    # training scheme
    parser.add_argument('--logdir', default="log/tcnp", help="log directory")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--l_alpha', default=0.9, type=float,
                        help="the weighting coefficient for trade-off between loss1 and loss2.")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--evaldir', default="eval/tcnp", help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=50, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=50, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default='data/tcnp.test.src',
                        help="input test data")
    parser.add_argument('--test2', default='data/tcnp.test.tgt',
                        help="output test data")
    parser.add_argument('--test_paraphrased', default='data/test_paraphrased_pair.txt',
                             help="test paraphrased pair dictionary")
    parser.add_argument('--ckpt', default="log/tcnp", help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="test/tcnp", help="test result dir")