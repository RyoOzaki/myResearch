#!/bin/bash

#==============================================
DSAE_STRUCTURE="39 20 10 6 3"
DSAE_PBHL_STRUCTURE="4 3"
SINGLE_DSAE_STRUCTURE="13 8 5 3"

#==============================================
# compress mcep using DSAE and DSAE-PBHL
#   feature_conv/dsae_all_speaker_25_10msec.npz (compress uninorm_concat_mfcc_all_speaker_25_10msec.npz)
#   feature_conv/dsae_spkind_all_speaker_25_10msec.npz (compress uninorm_spkind_concat_mfcc_all_speaker_25_10msec.npz)

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature_conv/uninorm_concat_mfcc_all_speaker_25_10msec.npz \
  --output_file feature_conv/dsae_all_speaker_25_10msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature_conv/uninorm_spkind_concat_mfcc_all_speaker_25_10msec.npz \
  --output_file feature_conv/dsae_spkind_all_speaker_25_10msec.npz \
  --structure ${DSAE_STRUCTURE}

#==============================================
#   feature_conv/dsae_pbhl_all_speaker_25_10msec.npz (compress uninorm_concat_mfcc_all_speaker_25_10msec.npz and speaker.npz)
#   feature_conv/dsae_pbhl_spkind_all_speaker_25_10msec.npz (compress uninorm_spkind_concat_mfcc_all_speaker_25_10msec.npz and speaker.npz)

python src/DSAE-PBHL/DSAE_PBHL_train.py \
  --train_data feature_conv/uninorm_concat_mfcc_all_speaker_25_10msec.npz \
  --speaker_id feature_conv/speaker.npz  \
  --output_file feature_conv/dsae_pbhl_all_speaker_25_10msec.npz \
  --structure ${DSAE_STRUCTURE} \
  --pb_structure ${DSAE_PBHL_STRUCTURE}

python src/DSAE-PBHL/DSAE_PBHL_train.py \
  --train_data feature_conv/uninorm_spkind_concat_mfcc_all_speaker_25_10msec.npz \
  --speaker_id feature_conv/speaker.npz  \
  --output_file feature_conv/dsae_pbhl_spkind_all_speaker_25_10msec.npz \
  --structure ${DSAE_STRUCTURE} \
  --pb_structure ${DSAE_PBHL_STRUCTURE}

#==============================================
#   feature_conv/dsae_pbhl_v2_all_speaker_25_10msec.npz (compress uninorm_concat_mfcc_all_speaker_25_10msec.npz and speaker.npz)
#   feature_conv/dsae_pbhl_v2_spkind_all_speaker_25_10msec.npz (compress uninorm_spkind_concat_mfcc_all_speaker_25_10msec.npz and speaker.npz)

python src/DSAE-PBHL/DSAE_PBHL_v2_train.py \
  --train_data feature_conv/uninorm_concat_mfcc_all_speaker_25_10msec.npz \
  --speaker_id feature_conv/speaker.npz  \
  --output_file feature_conv/dsae_pbhl_v2_all_speaker_25_10msec.npz \
  --structure ${DSAE_STRUCTURE} \
  --pb_structure ${DSAE_PBHL_STRUCTURE}

python src/DSAE-PBHL/DSAE_PBHL_v2_train.py \
  --train_data feature_conv/uninorm_spkind_concat_mfcc_all_speaker_25_10msec.npz \
  --speaker_id feature_conv/speaker.npz  \
  --output_file feature_conv/dsae_pbhl_v2_spkind_all_speaker_25_10msec.npz \
  --structure ${DSAE_STRUCTURE} \
  --pb_structure ${DSAE_PBHL_STRUCTURE}

#==============================================
#   feature_conv/dsae_speaker_H_25_10msec.npz (compress uninorm_mfcc_speaker_H_25_10msec.npz)
#   feature_conv/dsae_speaker_K_25_10msec.npz (compress uninorm_mfcc_speaker_K_25_10msec.npz)
#   feature_conv/dsae_speaker_M_25_10msec.npz (compress uninorm_mfcc_speaker_M_25_10msec.npz)
#   feature_conv/dsae_speaker_N_25_10msec.npz (compress uninorm_mfcc_speaker_N_25_10msec.npz)

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature_conv/uninorm_mfcc_speaker_H_25_10msec.npz \
  --output_file feature_conv/dsae_speaker_H_25_10msec.npz \
  --structure ${SINGLE_DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature_conv/uninorm_mfcc_speaker_K_25_10msec.npz \
  --output_file feature_conv/dsae_speaker_K_25_10msec.npz \
  --structure ${SINGLE_DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature_conv/uninorm_mfcc_speaker_M_25_10msec.npz \
  --output_file feature_conv/dsae_speaker_M_25_10msec.npz \
  --structure ${SINGLE_DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature_conv/uninorm_mfcc_speaker_N_25_10msec.npz \
  --output_file feature_conv/dsae_speaker_N_25_10msec.npz \
  --structure ${SINGLE_DSAE_STRUCTURE}
