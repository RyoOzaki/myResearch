#!/bin/bash

tester_all=("tester_1_1" "tester_2_1" "tester_3_1" "tester_4_1" "tester_5_1" "tester_6_1" "tester_1_2" "tester_2_2" "tester_3_2" "tester_4_2" "tester_5_2" "tester_6_2")

# make test
for tester in "${tester_all[@]}"
do
  echo ${tester}
  python src/Evaluate/make_language_model_test.py \
    --source_dir out_files_2/ \
    --cases \
      stargan_unigram \
      stargan_bigram \
      stargan_lstm \
    --num_of_test 10 \
    --speaker speaker_H \
    --output_dir test2_dirs/${tester}/test_2/test \
    --separate 1
done

# evaluate
python src/Evaluate/calc_mos_neary.py \
  --cases \
    stargan_unigram \
    stargan_bigram \
    stargan_lstm \
  --speaker speaker_H \
  --num_of_test 10 \
  --score_files \
    test2_result/1_1.csv \
    test2_result/1_2.csv \
    test2_result/2_1.csv \
    test2_result/2_2.csv \
    test2_result/4_1.csv \
    test2_result/4_2.csv \
  --load_npy
