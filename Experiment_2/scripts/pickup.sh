#!/bin/bash
# please run as follows
# $bash scripts/pickup.sh

cases=("stargan" "dsae_pbhl" "baseline")

pickup_sentences="aioi_uo_ie aue_ao"

for case in "${cases[@]}"
do
  echo ${case}
  python src/Evaluate/count_word.py \
    --word_num 10 \
    --raw_sentence parameters/${case}/LSTMLM/raw_sentences.npz \
    --output_dir figures/${case} \
    --speaker_id feature/speaker.npz

  python src/Evaluate/pickup.py \
    --raw_sentence parameters/${case}/LSTMLM/raw_sentences.npz \
    --speaker_id feature/speaker.npz \
    --output_file parameters/${case}/LSTMLM/pickuped_sentences.txt \
    --key_output parameters/key_of_pickuped_sentences.txt \
    --sentence ${pickup_sentences}

done

# doing case "topline"
