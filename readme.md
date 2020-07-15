# KEPN

Code for our AAAI2020 paper,
Integrating Linguistic Knowledge to Sentence Paraphrase Generation.

## Requirements
* python==3.6
* tensorflow==1.12.0

## Data Preprocessing
* STEP 1. vocab
```
python data_processing.py
```

* STEP 2. build paraphrase pairs
```
python prepro_dict.py
The file dict_synonym.txt is from "https://github.com/guotong1988/chinese_dictionary/blob/master/dict_synonym.txt"
```

## Train
```
python train.py
```
Refer to `hparams.py` for more details.

## Test
```
python test.py --ckpt log/tcnp
```
Some codes are based on this [repository](https://github.com/Kyubyong/transformer).

## Citation
If you find the code useful, please cite our paper.
```
@inproceedings{lin-2020-integrating,
    title = "Integrating Linguistic Knowledge to Sentence Paraphrase Generation",
    author = "Zibo Lin, Ziran Li, Ning Ding, Hai-Tao Zheng, Ying Shen, Wei Wang and Cong-Zhi Zhao. ",
    booktitle = "Proceedings of The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)",
    year = "2020",
}
```