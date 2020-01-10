#!/bin/bash

#==============================================
PCA_COMPONENTS="5"
# compress mcep using PCA
#   feature/pca_all_speaker_20msec.npz (compress mcep_all_speaker_20msec.npz)
#   feature/pca_speaker_H_20msec.npz (compress mcep_speaker_H_20msec.npz)
#   feature/pca_speaker_K_20msec.npz (compress mcep_speaker_K_20msec.npz)
#   feature/pca_speaker_M_20msec.npz (compress mcep_speaker_M_20msec.npz)
#   feature/pca_speaker_N_20msec.npz (compress mcep_speaker_N_20msec.npz)

python src/Compress/PCA_compress.py \
  --source_file feature/mcep_all_speaker_20msec.npz \
  --output_file feature/pca_all_speaker_20msec.npz \
  --n_components ${PCA_COMPONENTS}

python src/Compress/PCA_compress.py \
  --source_file feature/mcep_speaker_H_20msec.npz \
  --output_file feature/pca_speaker_H_20msec.npz \
  --n_components ${PCA_COMPONENTS}

python src/Compress/PCA_compress.py \
  --source_file feature/mcep_speaker_K_20msec.npz \
  --output_file feature/pca_speaker_K_20msec.npz \
  --n_components ${PCA_COMPONENTS}

python src/Compress/PCA_compress.py \
  --source_file feature/mcep_speaker_M_20msec.npz \
  --output_file feature/pca_speaker_M_20msec.npz \
  --n_components ${PCA_COMPONENTS}

python src/Compress/PCA_compress.py \
  --source_file feature/mcep_speaker_N_20msec.npz \
  --output_file feature/pca_speaker_N_20msec.npz \
  --n_components ${PCA_COMPONENTS}

#==============================================
DSAE_STRUCTURE="36 18 9 5"
DSAE_PBHL_STRUCTURE="4 3"

# compress mcep using DSAE and DSAE-PBHL
#   feature/dsae_all_speaker_20msec.npz (compress mcep_all_speaker_20msec.npz)
#   feature/dsae_pbhl_all_speaker_20msec.npz (compress mcep_all_speaker_20msec.npz and speaker.npz)
#   feature/dsae_pbhl_v2_all_speaker_20msec.npz (compress mcep_all_speaker_20msec.npz and speaker.npz)
#   feature/dsae_speaker_H_20msec.npz (compress mcep_speaker_H_20msec.npz)
#   feature/dsae_speaker_K_20msec.npz (compress mcep_speaker_K_20msec.npz)
#   feature/dsae_speaker_M_20msec.npz (compress mcep_speaker_M_20msec.npz)
#   feature/dsae_speaker_N_20msec.npz (compress mcep_speaker_N_20msec.npz)

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/mcep_all_speaker_20msec.npz \
  --output_file feature/dsae_all_speaker_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_PBHL_train.py \
  --train_data feature/mcep_all_speaker_20msec.npz \
  --speaker_id feature/speaker.npz  \
  --output_file feature/dsae_pbhl_all_speaker_20msec.npz \
  --structure ${DSAE_STRUCTURE} \
  --pb_structure ${DSAE_PBHL_STRUCTURE}

python src/DSAE-PBHL/DSAE_PBHL_v2_train.py \
  --train_data feature/mcep_all_speaker_20msec.npz \
  --speaker_id feature/speaker.npz  \
  --output_file feature/dsae_pbhl_v2_all_speaker_20msec.npz \
  --structure ${DSAE_STRUCTURE} \
  --pb_structure ${DSAE_PBHL_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/mcep_speaker_H_20msec.npz \
  --output_file feature/dsae_speaker_H_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/mcep_speaker_K_20msec.npz \
  --output_file feature/dsae_speaker_K_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/mcep_speaker_M_20msec.npz \
  --output_file feature/dsae_speaker_M_20msec.npz \
  --structure ${DSAE_STRUCTURE}

python src/DSAE-PBHL/DSAE_train.py \
  --train_data feature/mcep_speaker_N_20msec.npz \
  --output_file feature/dsae_speaker_N_20msec.npz \
  --structure ${DSAE_STRUCTURE}

#==============================================

# compress mcep using StarGAN-VC
#   feature/sgvc_all_speaker_20msec.npz (compress norm_with_f0_spkind_all_speaker_5msec.npz)
#   feature/sgvc_new_all_speaker_20msec.npz (compress norm_with_f0_spkind_all_speaker_5msec.npz)

# python src/StarGAN-VC/train_stargan-vc.py \
#   --train_data feature/norm_with_f0_spkind_all_speaker_5msec.npz \
#   --speaker_id feature/speaker.npz \
#   --output_file feature/sgvc_all_speaker_20msec.npz \
#   --batchsize 8 \
#   --epoch 6000 \
#   --gpu 1 \
#   --genpath feature/sgvc_all_speaker_20msec/snapshot/3000.gen.npz \
#   --clspath feature/sgvc_all_speaker_20msec/snapshot/3000.cls.npz \
#   --advdispath feature/sgvc_all_speaker_20msec/snapshot/3000.advdis.npz \
#   --epoch_start 3001

python src/StarGAN-VC/train_stargan-vc_new.py \
  --train_data feature/norm_with_f0_spkind_all_speaker_5msec.npz \
  --speaker_id feature/speaker.npz \
  --output_file feature/sgvc_new_all_speaker_20msec.npz \
  --batchsize 4 \
  --epoch 6000 \
  --gpu 1

python src/StarGAN-VC/train_stargan-vc_pp.py \
  --train_data feature/norm_with_f0_spkind_all_speaker_5msec.npz \
  --speaker_id feature/speaker.npz \
  --output_file feature/sgvc_pp_all_speaker_20msec.npz \
  --batchsize 8 \
  --epoch 6000 \
  --gpu 1
