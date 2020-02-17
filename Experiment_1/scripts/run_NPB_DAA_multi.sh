#!/bin/bash

#==============================================
# mfcc
#   feature/dsae_uninorm_concat_mfcc_all_speaker_20msec.npz
#   feature/dsae_pbhl_uninorm_concat_mfcc_all_speaker_20msec.npz

python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/compressed_dsae_mfcc_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_uninorm_concat_mfcc_all_speaker_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -s feature/speaker.npz \
  -l segmentation_result/dsae_uninorm_concat_mfcc_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_pbhl_uninorm_concat_mfcc_all_speaker_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -s feature/speaker.npz \
  -l segmentation_result/dsae_pbhl_uninorm_concat_mfcc_all_speaker_20msec

#==============================================
# mcep
#   feature/dsae_uninorm_mcep_all_speaker_20msec.npz
#   feature/dsae_pbhl_uninorm_mcep_all_speaker_20msec.npz

python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/compressed_dsae_mcep_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_uninorm_mcep_all_speaker_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -s feature/speaker.npz \
  -l segmentation_result/dsae_uninorm_mcep_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_pbhl_uninorm_mcep_all_speaker_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -s feature/speaker.npz \
  -l segmentation_result/dsae_pbhl_uninorm_mcep_all_speaker_20msec

#==============================================
# sgvc
#   feature/filt_sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz

python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/compressed_sgvc_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/filt_sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz \
  -p feature/phn_all_speaker_20msec.npz \
  -w feature/wrd_all_speaker_20msec.npz \
  -s feature/speaker.npz \
  -l segmentation_result/filt_sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec
