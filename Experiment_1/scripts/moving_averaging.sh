#!/bin/bash

#==============================================
# compress mcep using StarGAN-VC
#   feature/filt_sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz (compress sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz)

python src/Extractor/filtering.py \
  --source sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz \
  --output filt_sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz
