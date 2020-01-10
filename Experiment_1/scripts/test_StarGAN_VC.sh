#!/bin/bash

# python src/StarGAN-VC/test_StarGAN_VC.py \
#   --generator feature/sgvc_all_speaker/snapshot/4000.gen.npz \
#   --discriminator feature/sgvc_all_speaker/snapshot/4000.advdis.npz \
#   --classifier feature/sgvc_all_speaker/snapshot/4000.cls.npz \
#   --speaker_id feature/speaker.npz \
#   --mcep feature/mcep_all_speaker_5msec.npz \
#   --f0 feature/f0_all_speaker_5msec.npz \
#   --ap feature/ap_all_speaker_5msec.npz \
#   --mcep_norm_param feature/norm_with_f0_spkind_all_speaker_5msec/means.npz feature/norm_with_f0_spkind_all_speaker_5msec/stds.npz \
#   --logf0_norm_param feature/norm_with_f0_spkind_all_speaker_5msec/lf0_means.npz feature/norm_with_f0_spkind_all_speaker_5msec/lf0_stds.npz \
#   --output_dir test_sgvc

# python src/StarGAN-VC/test_StarGAN_VC.py \
#   --speaker_id feature/speaker.npz \
#   --mcep feature/mcep_all_speaker_5msec.npz \
#   --f0 feature/f0_all_speaker_5msec.npz \
#   --ap feature/ap_all_speaker_5msec.npz \
#   --mcep_norm_param feature/norm_with_f0_spkind_all_speaker_5msec/means.npz feature/norm_with_f0_spkind_all_speaker_5msec/stds.npz \
#   --logf0_norm_param feature/norm_with_f0_spkind_all_speaker_5msec/lf0_means.npz feature/norm_with_f0_spkind_all_speaker_5msec/lf0_stds.npz \
#   --snapshot_dir feature/sgvc_all_speaker_20msec/snapshot \
#   --snapshot_name 2400 \
#   --output_dir test_sgvc/ITER_2400

# python src/StarGAN-VC/test_StarGAN_VC.py \
#   --speaker_id feature/speaker.npz \
#   --mcep feature/mcep_all_speaker_5msec.npz \
#   --f0 feature/f0_all_speaker_5msec.npz \
#   --ap feature/ap_all_speaker_5msec.npz \
#   --frame_period 5E-3 \
#   --fftsize 2048 \
#   --mcep_norm_param feature/norm_with_f0_spkind_all_speaker_5msec/means.npz feature/norm_with_f0_spkind_all_speaker_5msec/stds.npz \
#   --logf0_norm_param feature/norm_with_f0_spkind_all_speaker_5msec/lf0_means.npz feature/norm_with_f0_spkind_all_speaker_5msec/lf0_stds.npz \
#   --snapshot_dir feature/sgvc_all_speaker_20msec/snapshot \
#   --snapshot_name 6000 \
#   --output_dir test_sgvc/ITER_6000

python src/StarGAN-VC/test_StarGAN_VC_new.py \
  --speaker_id feature/speaker.npz \
  --mcep feature/mcep_all_speaker_5msec.npz \
  --f0 feature/f0_all_speaker_5msec.npz \
  --ap feature/ap_all_speaker_5msec.npz \
  --frame_period 5E-3 \
  --fftsize 2048 \
  --mcep_norm_param feature/norm_with_f0_spkind_all_speaker_5msec/means.npz feature/norm_with_f0_spkind_all_speaker_5msec/stds.npz \
  --logf0_norm_param feature/norm_with_f0_spkind_all_speaker_5msec/lf0_means.npz feature/norm_with_f0_spkind_all_speaker_5msec/lf0_stds.npz \
  --snapshot_dir feature/sgvc_new_all_speaker_20msec/snapshot \
  --snapshot_name 1700 \
  --output_dir test_sgvc_new/ITER_1700
