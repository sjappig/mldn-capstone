#!/bin/bash

floyd run --gpu --env tensorflow-1.2:py2 --no-open --data sjappig/audioset_train/2:pp_data "python -m audiolabel.fit_and_predict /pp_data/train.h5 --test /pp_data/test.h5 --skip baseline --epochs 1500 --validation-size 0"

