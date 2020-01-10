#!/bin/bash

#==============================================
DSAE_STRUCTURE="39 20 10 6 3"
DSAE_PBHL_STRUCTURE="4 3"

# compress mcep using DSAE and DSAE-PBHL
#   feature/dsae_uninorm_concat_mfcc_all_speaker_20msec.npz (compress uninorm_concat_mfcc_all_speaker_20msec.npz)
#   feature/dsae_pbhl_uninorm_concat_mfcc_all_speaker_20msec.npz (compress uninorm_concat_mfcc_all_speaker_20msec.npz and speaker.npz)
#   feature/dsae_pbhl_v2_uninorm_concat_mfcc_all_speaker_20msec.npz (compress uninorm_concat_mfcc_all_speaker_20msec.npz and speaker.npz)

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_concat_mfcc_all_speaker_20msec.npz \
  --output_file feature/dsae_uninorm_concat_mfcc_all_speaker_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_PBHL_train.py \
  --train_data feature/uninorm_concat_mfcc_all_speaker_20msec.npz \
  --speaker_id feature/speaker.npz  \
  --output_file feature/dsae_pbhl_uninorm_concat_mfcc_all_speaker_20msec.npz \
  --structure ${DSAE_STRUCTURE} \
  --pb_structure ${DSAE_PBHL_STRUCTURE}

python src/DSAE-PBHL/DSAE_PBHL_v2_train.py \
  --train_data feature/uninorm_concat_mfcc_all_speaker_20msec.npz \
  --speaker_id feature/speaker.npz  \
  --output_file feature/dsae_pbhl_v2_uninorm_concat_mfcc_all_speaker_20msec.npz \
  --structure ${DSAE_STRUCTURE} \
  --pb_structure ${DSAE_PBHL_STRUCTURE}

#==============================================
DSAE_STRUCTURE="13 8 5 3"

# compress mcep using DSAE and DSAE-PBHL
#   feature/dsae_uninorm_mfcc_speaker_H_20msec.npz (compress uninorm_mfcc_speaker_H_20msec.npz)
#   feature/dsae_uninorm_mfcc_speaker_K_20msec.npz (compress uninorm_mfcc_speaker_K_20msec.npz)
#   feature/dsae_uninorm_mfcc_speaker_M_20msec.npz (compress uninorm_mfcc_speaker_M_20msec.npz)
#   feature/dsae_uninorm_mfcc_speaker_N_20msec.npz (compress uninorm_mfcc_speaker_N_20msec.npz)

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_mfcc_speaker_H_20msec.npz \
  --output_file feature/dsae_uninorm_mfcc_speaker_H_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_mfcc_speaker_K_20msec.npz \
  --output_file feature/dsae_uninorm_mfcc_speaker_K_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_mfcc_speaker_M_20msec.npz \
  --output_file feature/dsae_uninorm_mfcc_speaker_M_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/uninorm_mfcc_speaker_N_20msec.npz \
  --output_file feature/dsae_uninorm_mfcc_speaker_N_20msec.npz \
  --structure ${DSAE_STRUCTURE}

#==============================================

# compress mcep using StarGAN-VC
#   feature/sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz (compress gaunorm_with_f0_spkind_mcep_all_speaker_5msec.npz)
#   feature/sgvc_new_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz (compress gaunorm_with_f0_spkind_mcep_all_speaker_5msec.npz)

python src/StarGAN-VC/train_stargan-vc.py \
  --train_data feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec.npz \
  --speaker_id feature/speaker.npz \
  --output_file feature/sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz \
  --batchsize 4 \
  --epoch 6000 \
  --gpu 1
  # --genpath feature/sgvc_all_speaker_20msec/snapshot/3000.gen.npz \
  # --clspath feature/sgvc_all_speaker_20msec/snapshot/3000.cls.npz \
  # --advdispath feature/sgvc_all_speaker_20msec/snapshot/3000.advdis.npz \
  # --epoch_start 3001

python src/StarGAN-VC/train_stargan-vc_new.py \
  --train_data feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec.npz \
  --speaker_id feature/speaker.npz \
  --output_file feature/sgvc_new_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz \
  --batchsize 4 \
  --epoch 6000 \
  --gpu 1

# python src/StarGAN-VC/train_stargan-vc_pp.py \
#   --train_data feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec.npz \
#   --speaker_id feature/speaker.npz \
#   --output_file feature/sgvc_pp_all_speaker_20msec.npz \
#   --batchsize 8 \
#   --epoch 6000 \
#   --gpu 1
