#!/bin/bash

#==============================================
# compress mfcc using DSAE and DSAE-PBHL
#   feature/dsae_uninorm_concat_mfcc_all_speaker_20msec.npz (compress uninorm_concat_mfcc_all_speaker_20msec.npz)
#   feature/dsae_pbhl_uninorm_concat_mfcc_all_speaker_20msec.npz (compress uninorm_concat_mfcc_all_speaker_20msec.npz and speaker.npz)
DSAE_STRUCTURE="39 20 10 6 3"
DSAE_PBHL_STRUCTURE="4 3"

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_concat_mfcc_all_speaker_20msec.npz \
  --output feature/dsae_uninorm_concat_mfcc_all_speaker_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_PBHL_train.py \
  --train_data feature/uninorm_concat_mfcc_all_speaker_20msec.npz \
  --speaker_id feature/speaker.npz  \
  --output feature/dsae_pbhl_uninorm_concat_mfcc_all_speaker_20msec.npz \
  --structure ${DSAE_STRUCTURE} \
  --pb_structure ${DSAE_PBHL_STRUCTURE}

#==============================================
# compress mcep using DSAE and DSAE-PBHL
#   feature/dsae_uninorm_mcep_all_speaker_20msec.npz (compress uninorm_mcep_all_speaker_20msec.npz)
#   feature/dsae_pbhl_uninorm_mcep_all_speaker_20msec.npz (compress uninorm_mcep_all_speaker_20msec.npz and speaker.npz)
DSAE_STRUCTURE="36 18 9 5"
DSAE_PBHL_STRUCTURE="4 3"

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_mcep_all_speaker_20msec.npz \
  --output feature/dsae_uninorm_mcep_all_speaker_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_PBHL_train.py \
  --train_data feature/uninorm_mcep_all_speaker_20msec.npz \
  --speaker_id feature/speaker.npz  \
  --output feature/dsae_pbhl_uninorm_mcep_all_speaker_20msec.npz \
  --structure ${DSAE_STRUCTURE} \
  --pb_structure ${DSAE_PBHL_STRUCTURE}

#==============================================
# compress mfcc using DSAE and DSAE-PBHL (single speaker)
#   feature/dsae_uninorm_mfcc_speaker_H_20msec.npz (compress uninorm_mfcc_speaker_H_20msec.npz)
#   feature/dsae_uninorm_mfcc_speaker_K_20msec.npz (compress uninorm_mfcc_speaker_K_20msec.npz)
#   feature/dsae_uninorm_mfcc_speaker_M_20msec.npz (compress uninorm_mfcc_speaker_M_20msec.npz)
#   feature/dsae_uninorm_mfcc_speaker_N_20msec.npz (compress uninorm_mfcc_speaker_N_20msec.npz)
DSAE_STRUCTURE="13 8 5 3"

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_mfcc_speaker_H_20msec.npz \
  --output feature/dsae_uninorm_mfcc_speaker_H_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_mfcc_speaker_K_20msec.npz \
  --output feature/dsae_uninorm_mfcc_speaker_K_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_mfcc_speaker_M_20msec.npz \
  --output feature/dsae_uninorm_mfcc_speaker_M_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_mfcc_speaker_N_20msec.npz \
  --output feature/dsae_uninorm_mfcc_speaker_N_20msec.npz \
  --structure ${DSAE_STRUCTURE}

#==============================================
# compress mcep using StarGAN-VC
#   feature/sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz (compress gaunorm_with_f0_spkind_mcep_all_speaker_5msec.npz)

python src/StarGAN-VC/train_stargan-vc.py \
  --train_data feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec.npz \
  --speaker_id feature/speaker.npz \
  --output feature/sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz \
  --batchsize 8 \
  --epoch 2000 \
  --gpu 1
