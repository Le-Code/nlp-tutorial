# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Preprocess the iwslt 2016 datasets.
'''

#全文本位置:/data/234/UM-Corpus/data/Bilingual/allTex.txt


import os
import errno
import sentencepiece as spm
import re

import logging

logging.basicConfig(level=logging.CRITICAL)

if 0:
    """Load raw data -> Preprocessing -> Segmenting with sentencepice
    hp: hyperparams. argparse.
    """
    print('kaishi')
    if not os.path.exists('中英文bpe模型'):
        os.mkdir('中英文bpe模型')
    logging.info("# Train a joint BPE model with sentencepiece")
    #train参数是, unk id,bos id, eos id设置好即可.
    train = '--input=/data/234/UM-Corpus/data/Bilingual/Bi-Education.txt --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=中英文bpe模型/bpe --vocab_size={} \
             --model_type=bpe '.format(32000)
    spm.SentencePieceTrainer.Train(train)
    print('完成训练了!')
    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load("中英文bpe模型/bpe.model")
    print(sp.EncodeAsPieces('dsjafljdsl,我是一个人'))

if 0:
    sp = spm.SentencePieceProcessor()
    sp.Load("中英文bpe模型/bpe.model")
    print(sp.EncodeAsPieces('dsjafljdsl,我是一个人客结合线上线下大数据，线上通过运营商强大的数据挖掘能力，多方位精准锁定用户。线下通过场景大数据，依托智能硬件与大数据，汇集海量移动媒体和终端资源，为企业提供移动互联网精准营销与大数据应用服务'))
    _prepro = lambda x:  [line.strip() for line in open(x, 'r').read().split("\n")  ]
    prepro_train1 = _prepro('/data/234/UM-Corpus/data/Bilingual/Bi-Education.txt')

    def _segment_and_write(sents, fname):
        with open(fname, "w") as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                fout.write(" ".join(pieces) + "\n")
    _segment_and_write(prepro_train1, "chineseEnglishData.bpe")
    ##这个文件chineseEnglishData.bpe就是训练集了.

    #在根据奇偶行拆分batch训练即可.然后根据同目录里面的.vocab进行编码


    #然后得到的分词,进入对应的vocab里面去查询即可.

if 1:
    #把中英文bpe模型/bpe.vocab变成字典
    from collections import defaultdict
    dicA=defaultdict(int)
    with open('中英文bpe模型/bpe.vocab') as t:
        tmp=        t.readlines()
        for i in range(len(tmp)):

            a=tmp[i].strip('\n').split('\t')
            dicA[a[0]]=a[1]
    print(dicA)

    sp = spm.SentencePieceProcessor()
    sp.Load("中英文bpe模型/bpe.model")
    print(sp.EncodeAsPieces('dsjafljdsl,我是一个人客结合线上线下大数据，线上通过运营商强大的数据挖掘能力，多方位精准锁定用户。线下通过场景大数据，依托智能硬件与大数据，汇集海量移动媒体和终端资源，为企业提供移动互联网精准营销与大数据应用服务'))
    _prepro = lambda x: [line.strip() for line in open(x, 'r').read().split("\n")]
    prepro_train1 = _prepro('/data/234/UM-Corpus/data/Bilingual/Bi-Education.txt')


    def _segment_and_write(sents, fname):
        with open(fname, "w") as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                pieces=[str(dicA[i]) for i in pieces]
                fout.write(" ".join(pieces) + "\n")


    _segment_and_write(prepro_train1, "chineseEnglishDataCoded.bpe")
#跑完了,以后就用这个chineseEnglishDataCoded.bpe  这个编码后的进行学习了.