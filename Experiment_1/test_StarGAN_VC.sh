#!/bin/bash

python src/StarGAN-VC/test.py \
  --generator feature/sgvc_all_speaker/snapshot/4000.gen.npz \
  --discriminator feature/sgvc_all_speaker/snapshot/4000.advdis.npz \
  --classifier feature/sgvc_all_speaker/snapshot/4000.cls.npz \
  --speaker_id feature/speaker.npz \
  --mcep feature/mcep_all_speaker_5msec.npz \
  --f0 feature/f0_all_speaker_5msec.npz \
  --ap feature/ap_all_speaker_5msec.npz \
  --mcep_norm_param feature/norm_with_f0_spkind_all_speaker_5msec/means.npz feature/norm_with_f0_spkind_all_speaker_5msec/stds.npz \
  --logf0_norm_param feature/norm_with_f0_spkind_all_speaker_5msec/lf0_means.npz feature/norm_with_f0_spkind_all_speaker_5msec/lf0_stds.npz \
  --output_dir test_sgvc
