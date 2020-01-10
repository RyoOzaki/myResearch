#!/bin/bash

#==============================================
# 3 speaker dsae, dsae_pbhl, dsae_pbhl_v2
python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/compressed_dsae_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_uninorm_concat_mfcc_all_speaker_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -l segmentation_result/dsae_uninorm_concat_mfcc_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_pbhl_uninorm_concat_mfcc_all_speaker_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -l segmentation_result/dsae_pbhl_uninorm_concat_mfcc_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_pbhl_v2_uninorm_concat_mfcc_all_speaker_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -l segmentation_result/dsae_pbhl_v2_uninorm_concat_mfcc_all_speaker_20msec

#==============================================
# 3 speaker sgvc, sgvc_new
python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/compressed_sgvc_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -l segmentation_result/sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/sgvc_new_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -l segmentation_result/sgvc_new_gaunorm_with_f0_spkind_mcep_all_speaker_20msec

#==============================================
# single speaker dsae
python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/compressed_dsae_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_uninorm_mfcc_speaker_H_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -l segmentation_result/dsae_uninorm_mfcc_speaker_H_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_uninorm_mfcc_speaker_K_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -l segmentation_result/dsae_uninorm_mfcc_speaker_K_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_uninorm_mfcc_speaker_M_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -l segmentation_result/dsae_uninorm_mfcc_speaker_M_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_uninorm_mfcc_speaker_N_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -l segmentation_result/dsae_uninorm_mfcc_speaker_N_20msec
