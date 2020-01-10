#!/bin/bash

#==============================================
# DSAE

python src/NPB-DAA/unroll_default_config.py \
  --default_config conventional_scripts/default_configs/defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature_conv/dsae_all_speaker_25_10msec.npz \
  -p feature_conv/phn_all_speaker_25_10msec.npz \
  -w feature_conv/wrd_all_speaker_25_10msec.npz \
  -l segmentation_result_conv/dsae_all_speaker_25_10msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature_conv/dsae_spkind_all_speaker_25_10msec.npz \
  -p feature_conv/phn_all_speaker_25_10msec.npz \
  -w feature_conv/wrd_all_speaker_25_10msec.npz \
  -l segmentation_result_conv/dsae_spkind_all_speaker_25_10msec

#==============================================
# DSAE-PBHL

python src/NPB-DAA/unroll_default_config.py \
  --default_config conventional_scripts/default_configs/defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature_conv/dsae_pbhl_all_speaker_25_10msec.npz \
  -p feature_conv/phn_all_speaker_25_10msec.npz \
  -w feature_conv/wrd_all_speaker_25_10msec.npz \
  -l segmentation_result_conv/dsae_pbhl_all_speaker_25_10msec 

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature_conv/dsae_pbhl_spkind_all_speaker_25_10msec.npz \
  -p feature_conv/phn_all_speaker_25_10msec.npz \
  -w feature_conv/wrd_all_speaker_25_10msec.npz \
  -l segmentation_result_conv/dsae_pbhl_spkind_all_speaker_25_10msec

#==============================================
# DSAE-PBHL v2

python src/NPB-DAA/unroll_default_config.py \
  --default_config conventional_scripts/default_configs/defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature_conv/dsae_pbhl_v2_all_speaker_25_10msec.npz \
  -p feature_conv/phn_all_speaker_25_10msec.npz \
  -w feature_conv/wrd_all_speaker_25_10msec.npz \
  -l segmentation_result_conv/dsae_pbhl_v2_all_speaker_25_10msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature_conv/dsae_pbhl_v2_spkind_all_speaker_25_10msec.npz \
  -p feature_conv/phn_all_speaker_25_10msec.npz \
  -w feature_conv/wrd_all_speaker_25_10msec.npz \
  -l segmentation_result_conv/dsae_pbhl_v2_spkind_all_speaker_25_10msec

#==============================================
# Single speaker

python src/NPB-DAA/unroll_default_config.py \
  --default_config conventional_scripts/default_configs/defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature_conv/dsae_speaker_H_25_10msec.npz \
  -p feature_conv/phn_all_speaker_25_10msec.npz \
  -w feature_conv/wrd_all_speaker_25_10msec.npz \
  -l segmentation_result_conv/dsae_speaker_H_25_10msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature_conv/dsae_speaker_K_25_10msec.npz \
  -p feature_conv/phn_all_speaker_25_10msec.npz \
  -w feature_conv/wrd_all_speaker_25_10msec.npz \
  -l segmentation_result_conv/dsae_speaker_K_25_10msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature_conv/dsae_speaker_M_25_10msec.npz \
  -p feature_conv/phn_all_speaker_25_10msec.npz \
  -w feature_conv/wrd_all_speaker_25_10msec.npz \
  -l segmentation_result_conv/dsae_speaker_M_25_10msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature_conv/dsae_speaker_N_25_10msec.npz \
  -p feature_conv/phn_all_speaker_25_10msec.npz \
  -w feature_conv/wrd_all_speaker_25_10msec.npz \
  -l segmentation_result_conv/dsae_speaker_N_25_10msec
