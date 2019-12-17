#!/bin/bash

python src/NPB-DAA/unroll_default_config.py --default_config default_configs/compressed_defaults.config
sh src/NPB-DAA/watchdog_runner.sh -t feature/pca_all_speaker.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/pca_all_speaker
sh src/NPB-DAA/watchdog_runner.sh -t feature/pca_speaker_H.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/pca_speaker_H
sh src/NPB-DAA/watchdog_runner.sh -t feature/pca_speaker_K.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/pca_speaker_K
sh src/NPB-DAA/watchdog_runner.sh -t feature/pca_speaker_M.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/pca_speaker_M
sh src/NPB-DAA/watchdog_runner.sh -t feature/pca_speaker_N.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/pca_speaker_N

python src/NPB-DAA/unroll_default_config.py --default_config default_configs/compressed_defaults.config
sh src/NPB-DAA/watchdog_runner.sh -t feature/dsae_all_speaker.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_all_speaker
sh src/NPB-DAA/watchdog_runner.sh -t feature/dsae_pbhl_all_speaker.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_pbhl_all_speaker
sh src/NPB-DAA/watchdog_runner.sh -t feature/dsae_pbhl_v2_all_speaker.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_pbhl_v2_all_speaker
sh src/NPB-DAA/watchdog_runner.sh -t feature/dsae_speaker_H.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_speaker_H
sh src/NPB-DAA/watchdog_runner.sh -t feature/dsae_speaker_K.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_speaker_K
sh src/NPB-DAA/watchdog_runner.sh -t feature/dsae_speaker_M.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_speaker_M
sh src/NPB-DAA/watchdog_runner.sh -t feature/dsae_speaker_N.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_speaker_N

python src/NPB-DAA/unroll_default_config.py --default_config default_configs/compressed_defaults.config
sh src/NPB-DAA/watchdog_runner.sh -t feature/sgvc_all_speaker.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/sgvc_all_speaker

python src/NPB-DAA/unroll_default_config.py --default_config default_configs/mcep_defaults.config
sh src/NPB-DAA/watchdog_runner.sh -t feature/mcep_all_speaker_20msec.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/mcep_all_speaker
sh src/NPB-DAA/watchdog_runner.sh -t feature/mcep_speaker_H.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/mcep_speaker_H
sh src/NPB-DAA/watchdog_runner.sh -t feature/mcep_speaker_K.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/mcep_speaker_K
sh src/NPB-DAA/watchdog_runner.sh -t feature/mcep_speaker_M.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/mcep_speaker_M
sh src/NPB-DAA/watchdog_runner.sh -t feature/mcep_speaker_N.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/mcep_speaker_N
