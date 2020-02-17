#!/bin/bash

python src/Feature_analysis/plot_pca.py \
  --phn feature/phn_all_speaker_20msec.npz \
  --speaker_id feature/speaker.npz \
  --phn_list s a e i o u \
  --not_show s \
  --source feature/dsae_uninorm_concat_mfcc_all_speaker_20msec.npz \
  --alpha 0.5 \
  --output_dir figure/DSAE

python src/Feature_analysis/plot_pca.py \
  --phn feature/phn_all_speaker_20msec.npz \
  --speaker_id feature/speaker.npz \
  --phn_list s a e i o u \
  --not_show s \
  --source feature/dsae_pbhl_uninorm_concat_mfcc_all_speaker_20msec.npz \
  --alpha 0.5 \
  --output_dir figure/DSAE_PBHL

python src/Feature_analysis/plot_pca.py \
  --phn feature/phn_all_speaker_20msec.npz \
  --speaker_id feature/speaker.npz \
  --phn_list s a e i o u \
  --not_show s \
  --source feature/filt_sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec.npz \
  --alpha 0.5 \
  --output_dir figure/StarGAN_VC
