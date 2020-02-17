#!/bin/bash

tester_all=("tester_1" "tester_2" "tester_3" "tester_4" "tester_5" "tester_6")

# make test
for tester in "${tester_all[@]}"
do
  echo ${tester}
  python src/Evaluate/make_transcription_test.py \
    --source_dir out_files/ \
    --cases \
      stargan \
      dsae_pbhl \
      topline \
      baseline \
    --speaker_id feature/speaker.npz \
    --key_of_sentences parameters/key_of_pickuped_sentences.txt \
    --output_dir test_dirs/${tester}/test_1/ \
    --separate 2
done

# evaluate
python src/Evaluate/calc_cer.py \
  --cases \
    stargan \
    dsae_pbhl \
    topline \
    baseline \
  --speaker_id feature/speaker.npz \
  --sentences aioiuoie aueao \
  --transcriptions \
    test1_result/1_1.csv \
    test1_result/1_2.csv \
  --transcriptions \
    test1_result/2_1.csv \
    test1_result/2_2.csv \
  --transcriptions \
    test1_result/4_1.csv \
    test1_result/4_2.csv \
  --load_npy
