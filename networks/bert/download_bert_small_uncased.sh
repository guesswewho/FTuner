#!/bin/bash -e
mkdir -p bert_small_uncased

wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-small-uncased-pytorch_model.bin \
     -O bert_small_uncased/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-small-uncased-vocab.txt \
     -O bert_small_uncased/vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-small-uncased-config.json \
     -O bert_small_uncased/config.json
