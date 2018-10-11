#!/bin/bash

python build-tagger.py sents.train model-file
python run-tagger.py sents.test model-file sents.out
python eval.py sents.out sents.answer