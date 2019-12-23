#!/bin/bash

#==============================================
python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/compressed_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/pca_all_speaker_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/pca_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/pca_speaker_H_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/pca_speaker_H_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/pca_speaker_K_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/pca_speaker_K_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/pca_speaker_M_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/pca_speaker_M_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/pca_speaker_N_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/pca_speaker_N_20msec

#==============================================
python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/compressed_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_all_speaker_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/dsae_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_pbhl_all_speaker_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/dsae_pbhl_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_pbhl_v2_all_speaker_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/dsae_pbhl_v2_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_speaker_H_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/dsae_speaker_H_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_speaker_K_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/dsae_speaker_K_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_speaker_M_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/dsae_speaker_M_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/dsae_speaker_N_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/dsae_speaker_N_20msec

#==============================================
python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/compressed_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/sgvc_all_speaker_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/sgvc_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/sgvc_new_all_speaker_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/sgvc_new_all_speaker_20msec

#==============================================
python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/mcep_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/mcep_all_speaker_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/mcep_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/mcep_speaker_H_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/mcep_speaker_H_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/mcep_speaker_K_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/mcep_speaker_K_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/mcep_speaker_M_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/mcep_speaker_M_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/mcep_speaker_N_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/mcep_speaker_N_20msec

#==============================================
python src/NPB-DAA/unroll_default_config.py \
  --default_config default_configs/mcep_defaults.config

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/norm_all_speaker_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/norm_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/norm_spkind_all_speaker_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/norm_spkind_all_speaker_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/norm_speaker_H_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/norm_speaker_H_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/norm_speaker_K_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/norm_speaker_K_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/norm_speaker_M_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/norm_speaker_M_20msec

sh src/NPB-DAA/watchdog_runner.sh \
  -t feature/norm_speaker_N_20msec.npz \
  -p feature/phn.npz \
  -w feature/wrd.npz \
  -l segmentation_result/norm_speaker_N_20msec
