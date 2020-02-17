#!/bin/bash

#==============================================
# single speaker mfcc
#   feature/dsae_uninorm_mfcc_speaker_H_20msec.npz
#   feature/dsae_uninorm_mfcc_speaker_K_20msec.npz
#   feature/dsae_uninorm_mfcc_speaker_M_20msec.npz
#   feature/dsae_uninorm_mfcc_speaker_N_20msec.npz

python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/compressed_dsae_mfcc_defaults.config

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
