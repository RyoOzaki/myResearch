#!/bin/bash

speakers=("speaker_H" "speaker_K" "speaker_M" "speaker_N")

# stargan
param_dir="parameters/stargan"
for tspk in "${speakers[@]}"
do
  echo "stargan:${tspk}"

  python src/stargan_vocoder.py \
    --samplerate 48000 \
    --fftsize 2048 \
    --frame_period 5E-3 \
    --sentences_file ${param_dir}/LSTMLM/sentences.npz \
    --letter_num 10 \
    --letter_stateseq ${param_dir}/NPBDAA/results/letter_stateseq.npz \
    --ap feature/ap_all_speaker_5msec.npz \
    --f0 feature/f0_all_speaker_5msec.npz \
    --parameter ${param_dir}/NPBDAA/parameters/ITR_0100.npz \
    --LM LSTM \
    --LSTM_model ${param_dir}/LSTMLM/model.h5 \
    --speaker_id feature/speaker.npz \
    --mcep_norm_param \
      feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/means.npz \
      feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/stds.npz \
    --target_speaker ${tspk} \
    --stargan_generator ${param_dir}/StarGANVC/2000.gen.npz \
    --output_prefix out_files/stargan/${tspk}/out \
    --mode ML \
    --flat_f0 \
    --flat_ap \
    @${param_dir}/LSTMLM/pickuped_sentences.txt

done

# DSAE-PBHL
param_dir="parameters/dsae_pbhl"
for tspk in "${speakers[@]}"
do
  echo "dsae_pbhl:${tspk}"
  python src/dsaepbhl_vocoder.py \
    --samplerate 48000 \
    --fftsize 2048 \
    --frame_period 20E-3 \
    --sentences_file ${param_dir}/LSTMLM/sentences.npz \
    --letter_num 10 \
    --letter_stateseq ${param_dir}/NPBDAA/results/letter_stateseq.npz \
    --ap feature/ap_all_speaker_20msec.npz \
    --f0 feature/f0_all_speaker_20msec.npz \
    --parameter ${param_dir}/NPBDAA/parameters/ITR_0100.npz \
    --LM LSTM \
    --LSTM_model ${param_dir}/LSTMLM/model.h5 \
    --speaker_id feature/speaker.npz \
    --mcep_norm_param \
      feature/uninorm_mcep_all_speaker_20msec/min.npy \
      feature/uninorm_mcep_all_speaker_20msec/max.npy \
    --target_speaker ${tspk} \
    --dsae_param feature/dsae_pbhl_uninorm_mcep_all_speaker_20msec/dsae.npz \
    --pb_param feature/dsae_pbhl_uninorm_mcep_all_speaker_20msec/pb_means.npz \
    --output_prefix out_files/dsae_pbhl/${tspk}/out \
    --mode ML \
    --flat_f0 \
    --flat_ap \
    @${param_dir}/LSTMLM/pickuped_sentences.txt
done

for tspk in "${speakers[@]}"
do
  echo "topline:${tspk}"

  python src/topline_vocoder.py \
    --samplerate 48000 \
    --fftsize 2048 \
    --frame_period 20E-3 \
    --ap feature/ap_all_speaker_20msec.npz \
    --f0 feature/f0_all_speaker_20msec.npz \
    --mcep feature/mcep_all_speaker_20msec.npz \
    --phn feature/phn_all_speaker_20msec.npz \
    --speaker_id feature/speaker.npz \
    --target_speaker ${tspk} \
    --output_prefix out_files/topline/${tspk}/out \
    --mode ML \
    --flat_f0 \
    --flat_ap \
    --key_of_pickuped_sentences @parameters/key_of_pickuped_sentences.txt

done

param_dir="parameters/baseline"
for tspk in "${speakers[@]}"
do
  echo "baseline:${tspk}"

  python src/baseline_vocoder.py \
    --samplerate 48000 \
    --fftsize 2048 \
    --frame_period 20E-3 \
    --sentences_file ${param_dir}/LSTMLM/sentences.npz \
    --letter_num 10 \
    --letter_stateseq ${param_dir}/NPBDAA/results/letter_stateseq.npz \
    --parameter ${param_dir}/NPBDAA/parameters/ITR_0100.npz \
    --ap feature/ap_all_speaker_20msec.npz \
    --f0 feature/f0_all_speaker_20msec.npz \
    --mcep feature/mcep_all_speaker_20msec.npz \
    --LM LSTM \
    --LSTM_model ${param_dir}/LSTMLM/model.h5 \
    --speaker_id feature/speaker.npz \
    --target_speaker ${tspk} \
    --output_prefix out_files/baseline/${tspk}/out \
    --mode ML \
    --flat_f0 \
    --flat_ap \
    @${param_dir}/LSTMLM/pickuped_sentences.txt
done
