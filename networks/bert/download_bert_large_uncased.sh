#!/bin/bash -e
mkdir -p bert_large_uncased

wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin \
     -O bert_large_uncased/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt \
     -O bert_large_uncased/vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json \
     -O bert_large_uncased/config.json
