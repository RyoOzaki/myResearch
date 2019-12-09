#!/bin/bash

python src/NPB-DAA/unroll_default_config.py --default_config default_configs/compressed_defaults.config
sh src/NPB-DAA/runner.sh -t feature/dsae_all_speaker.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_all_speaker
sh src/NPB-DAA/runner.sh -t feature/dsae_pbhl_all_speaker.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_all_speaker
sh src/NPB-DAA/runner.sh -t feature/dsae_speaker_H.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_speaker_H
sh src/NPB-DAA/runner.sh -t feature/dsae_speaker_K.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_speaker_K
sh src/NPB-DAA/runner.sh -t feature/dsae_speaker_M.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_speaker_M
sh src/NPB-DAA/runner.sh -t feature/dsae_speaker_N.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/dsae_speaker_N

python src/NPB-DAA/unroll_default_config.py --default_config default_configs/mcep_defaults.config
sh src/NPB-DAA/runner.sh -t feature/mcep_all_speaker.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/mcep_all_speaker
sh src/NPB-DAA/runner.sh -t feature/mcep_speaker_H.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/mcep_speaker_H
sh src/NPB-DAA/runner.sh -t feature/mcep_speaker_K.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/mcep_speaker_K
sh src/NPB-DAA/runner.sh -t feature/mcep_speaker_M.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/mcep_speaker_M
sh src/NPB-DAA/runner.sh -t feature/mcep_speaker_N.npz -p feature/phn.npz -w feature/wrd.npz -l segmentation_result/mcep_speaker_N
