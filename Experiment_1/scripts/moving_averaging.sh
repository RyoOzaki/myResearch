#!/bin/bash

#==============================================
# filtering mcep using StarGAN-VC
#   feature/filt_sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz (filtering sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz)

python src/Extractor/filtering.py \
  --source feature/sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz \
  --output feature/filt_sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz
