#!/bin/bash
EXP1_feature="../Experiment_1/feature/"

cp -r ${EXP1_feature}/gaunorm_with_f0_spkind_mcep_all_speaker_5msec feature/
cp ${EXP1_feature}/speaker.npz feature/
cp ${EXP1_feature}/ap_all_speaker_5msec.npz feature/
cp ${EXP1_feature}/f0_all_speaker_5msec.npz feature
cp ${EXP1_feature}/mcep_all_speaker_5msec.npz feature

cp ${EXP1_feature}/ap_all_speaker_20msec.npz feature/
cp ${EXP1_feature}/f0_all_speaker_20msec.npz feature/
cp ${EXP1_feature}/mcep_all_speaker_20msec.npz feature

cp -r ${EXP1_feature}/uninorm_mcep_all_speaker_20msec feature/
cp -r ${EXP1_feature}/dsae_uninorm_mcep_all_speaker_20msec feature/
cp -r ${EXP1_feature}/dsae_pbhl_uninorm_mcep_all_speaker_20msec feature/

sgvc_label="sgvc_gaunorm_with_f0_spkind_mcep_all_speaker_20msec"
using_iter="2000"
mkdir -p parameters/stargan/StarGANVC
cp ../Experiment_1/feature/${sgvc_label}/snapshot/${using_iter}.gen.npz parameters/stargan/StarGANVC/
