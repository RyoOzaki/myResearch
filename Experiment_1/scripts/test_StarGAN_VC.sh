#!/bin/bash

# python src/StarGAN-VC/test_StarGAN_VC.py \
#   --speaker_id feature/speaker.npz \
#   --mcep feature/mcep_all_speaker_5msec.npz \
#   --f0 feature/f0_all_speaker_5msec.npz \
#   --ap feature/ap_all_speaker_5msec.npz \
#   --frame_period 5E-3 \
#   --fftsize 2048 \
#   --mcep_norm_param feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/means.npz feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/stds.npz \
#   --logf0_norm_param feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/lf0_means.npz feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/lf0_stds.npz \
#   --snapshot_dir feature_tmp/sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec/snapshot \
#   --snapshot_name 2000 \
#   --output_dir test_sgvc/ITER_2000

python src/StarGAN-VC/test_StarGAN_VC.py \
  --speaker_id feature/speaker.npz \
  --mcep feature/mcep_all_speaker_5msec.npz \
  --f0 feature/f0_all_speaker_5msec.npz \
  --ap feature/ap_all_speaker_5msec.npz \
  --frame_period 5E-3 \
  --fftsize 2048 \
  --mcep_norm_param feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/means.npz feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/stds.npz \
  --logf0_norm_param feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/lf0_means.npz feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/lf0_stds.npz \
  --snapshot_dir feature_tmp/sgvc_new_gaunorm_with_f0_spkind_mcep_all_speaker_20msec/snapshot \
  --snapshot_name 400 \
  --output_dir test_sgvc_new/ITER_0400
