#!/bin/bash

#==============================================
# normalize
# output
#   feature_conv/uninorm_concat_mfcc_all_speaker_25_10msec.npz (normalize concat_mfcc_all_speaker_25_10msec.npz)
#   feature_conv/uninorm_spkind_concat_mfcc_all_speaker_25_10msec.npz (normalize concat_mfcc_all_speaker_25_10msec.npz) # speaker individual normalization
python src/Normalize/uniform_normalize.py \
  --source_file feature_conv/concat_mfcc_all_speaker_25_10msec.npz \
  --output_file feature_conv/uninorm_concat_mfcc_all_speaker_25_10msec.npz \
  --min_value -1.0 \
  --max_value  1.0

python src/Normalize/uniform_normalize.py \
  --source_file feature_conv/concat_mfcc_all_speaker_25_10msec.npz \
  --output_file feature_conv/uninorm_spkind_concat_mfcc_all_speaker_25_10msec.npz \
  --speaker_id feature_conv/speaker.npz \
  --min_value -1.0 \
  --max_value  1.0

#==============================================
# normalize
# output
#   feature_conv/uninorm_mfcc_speaker_H_25_10msec.npz (normalize mfcc_speaker_H_25_10msec.npz)
#   feature_conv/uninorm_mfcc_speaker_K_25_10msec.npz (normalize mfcc_speaker_K_25_10msec.npz)
#   feature_conv/uninorm_mfcc_speaker_M_25_10msec.npz (normalize mfcc_speaker_M_25_10msec.npz)
#   feature_conv/uninorm_mfcc_speaker_N_25_10msec.npz (normalize mfcc_speaker_N_25_10msec.npz)
python src/Normalize/uniform_normalize.py \
  --source_file feature_conv/mfcc_speaker_H_25_10msec.npz \
  --output_file feature_conv/uninorm_mfcc_speaker_H_25_10msec.npz \
  --min_value -1.0 \
  --max_value  1.0

python src/Normalize/uniform_normalize.py \
  --source_file feature_conv/mfcc_speaker_K_25_10msec.npz \
  --output_file feature_conv/uninorm_mfcc_speaker_K_25_10msec.npz \
  --min_value -1.0 \
  --max_value  1.0

python src/Normalize/uniform_normalize.py \
  --source_file feature_conv/mfcc_speaker_M_25_10msec.npz \
  --output_file feature_conv/uninorm_mfcc_speaker_M_25_10msec.npz \
  --min_value -1.0 \
  --max_value  1.0

python src/Normalize/uniform_normalize.py \
  --source_file feature_conv/mfcc_speaker_N_25_10msec.npz \
  --output_file feature_conv/uninorm_mfcc_speaker_N_25_10msec.npz \
  --min_value -1.0 \
  --max_value  1.0
