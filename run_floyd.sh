#!/bin/bash

floyd run --gpu --env tensorflow-1.2:py2 --no-open --data sjappig/audioset_train/1:pp_data "python -m audiolabel.fit_and_predict /pp_data/train.h5 --N 1000 --skip baseline"

