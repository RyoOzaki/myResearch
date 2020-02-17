#!/bin/bash

cases=("stargan" "dsae_pbhl" "baseline")
# cases=("stargan")

for case in "${cases[@]}"
do

  echo ${case}

  python src/LSTMLM/convert.py \
    --word_num 10 \
    --output parameters/${case}/LSTMLM/sentences.npz \
    --word_stateseq parameters/${case}/NPBDAA/results/word_stateseq.npz \
    --word_durations parameters/${case}/NPBDAA/results/word_durations.npz

  python src/LSTMLM/LSTMLM_train.py \
    --model parameters/${case}/LSTMLM/model.h5 \
    --train_data parameters/${case}/LSTMLM/sentences.npz \
    --train_iter 1000 \
    --hidden_node 128 \
    --init_model

done
