#!/bin/bash

#==============================================
# normalize
# output
#   feature/norm_all_speaker_20msec.npz (normalize mcep_all_speaker_20msec.npz)
#   feature/norm_spkind_all_speaker_20msec.npz (normalize mcep_all_speaker_20msec.npz) # speaker individual normalization
python src/Normalize/gaussian_normalize.py \
  --source_file feature/mcep_all_speaker_20msec.npz \
  --output_file feature/norm_all_speaker_20msec.npz

python src/Normalize/gaussian_normalize.py \
  --source_file feature/mcep_all_speaker_20msec.npz \
  --output_file feature/norm_spkind_all_speaker_20msec.npz \
  --speaker_id feature/speaker.npz

#==============================================
# normalize
# output
#   feature/norm_speaker_H_20msec.npz (normalize mcep_speaker_H_20msec.npz)
#   feature/norm_speaker_K_20msec.npz (normalize mcep_speaker_K_20msec.npz)
#   feature/norm_speaker_M_20msec.npz (normalize mcep_speaker_M_20msec.npz)
#   feature/norm_speaker_N_20msec.npz (normalize mcep_speaker_N_20msec.npz)
python src/Normalize/gaussian_normalize.py \
  --source_file feature/mcep_speaker_H_20msec.npz \
  --output_file feature/norm_speaker_H_20msec.npz

python src/Normalize/gaussian_normalize.py \
  --source_file feature/mcep_speaker_K_20msec.npz \
  --output_file feature/norm_speaker_K_20msec.npz

python src/Normalize/gaussian_normalize.py \
  --source_file feature/mcep_speaker_M_20msec.npz \
  --output_file feature/norm_speaker_M_20msec.npz

python src/Normalize/gaussian_normalize.py \
  --source_file feature/mcep_speaker_N_20msec.npz \
  --output_file feature/norm_speaker_N_20msec.npz

#==============================================
# normalize
# output
#   feature/norm_with_f0_spkind_all_speaker_5msec.npz (normalize mcep_all_speaker_5msec.npz) # speaker individual normalization
python src/Normalize/gaussian_normalize_with_f0.py \
  --source_file feature/mcep_all_speaker_5msec.npz \
  --source_f0 feature/f0_all_speaker_5msec.npz \
  --output_file feature/norm_with_f0_spkind_all_speaker_5msec.npz \
  --speaker_id feature/speaker.npz
