#!/bin/bash

#==============================================
# normalize
# output
#   feature/uninorm_concat_mfcc_all_speaker_20msec.npz (normalize concat_mfcc_all_speaker_20msec.npz)
python src/Normalize/uniform_normalize.py \
  --source feature/concat_mfcc_all_speaker_20msec.npz \
  --output feature/uninorm_concat_mfcc_all_speaker_20msec.npz \
  --min_value -1.0 \
  --max_value  1.0

# ==============================================
# normalize
# output
#   feature/uninorm_mcep_all_speaker_20msec.npz (normalize mcep_all_speaker_20msec.npz)
python src/Normalize/uniform_normalize.py \
  --source feature/mcep_all_speaker_20msec.npz \
  --output feature/uninorm_mcep_all_speaker_20msec.npz \
  --min_value -1.0 \
  --max_value  1.0

#==============================================
# normalize
# output
#   feature/uninorm_mfcc_speaker_H_20msec.npz (normalize mfcc_speaker_H_20msec.npz)
#   feature/uninorm_mfcc_speaker_K_20msec.npz (normalize mfcc_speaker_K_20msec.npz)
#   feature/uninorm_mfcc_speaker_M_20msec.npz (normalize mfcc_speaker_M_20msec.npz)
#   feature/uninorm_mfcc_speaker_N_20msec.npz (normalize mfcc_speaker_N_20msec.npz)
python src/Normalize/uniform_normalize.py \
  --source feature/mfcc_speaker_H_20msec.npz \
  --output feature/uninorm_mfcc_speaker_H_20msec.npz \
  --min_value -1.0 \
  --max_value  1.0

python src/Normalize/uniform_normalize.py \
  --source feature/mfcc_speaker_K_20msec.npz \
  --output feature/uninorm_mfcc_speaker_K_20msec.npz \
  --min_value -1.0 \
  --max_value  1.0

python src/Normalize/uniform_normalize.py \
  --source feature/mfcc_speaker_M_20msec.npz \
  --output feature/uninorm_mfcc_speaker_M_20msec.npz \
  --min_value -1.0 \
  --max_value  1.0

python src/Normalize/uniform_normalize.py \
  --source feature/mfcc_speaker_N_20msec.npz \
  --output feature/uninorm_mfcc_speaker_N_20msec.npz \
  --min_value -1.0 \
  --max_value  1.0

# #==============================================
# normalize
# output
#   feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec.npz (normalize mcep_all_speaker_5msec.npz) # speaker individual normalization
python src/Normalize/gaussian_normalize_with_f0.py \
  --source feature/mcep_all_speaker_5msec.npz \
  --source_f0 feature/f0_all_speaker_5msec.npz \
  --output feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec.npz \
  --speaker_id feature/speaker.npz
