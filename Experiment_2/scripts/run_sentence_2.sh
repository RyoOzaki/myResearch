#!/bin/bash

speakers=("speaker_H" "speaker_K" "speaker_M" "speaker_N")
# speakers=("speaker_H")
size="10"

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
    --LM Unigram \
    --speaker_id feature/speaker.npz \
    --mcep_norm_param \
      feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/means.npz \
      feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/stds.npz \
    --target_speaker ${tspk} \
    --stargan_generator ${param_dir}/StarGANVC/2000.gen.npz \
    --output_prefix out_files_2/stargan_unigram/${tspk}/out \
    --mode ML \
    --flat_f0 \
    --flat_ap \
    --size ${size} \
    --unique
done

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
    --LM Bigram \
    --speaker_id feature/speaker.npz \
    --mcep_norm_param \
      feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/means.npz \
      feature/gaunorm_with_f0_spkind_mcep_all_speaker_5msec/stds.npz \
    --target_speaker ${tspk} \
    --stargan_generator ${param_dir}/StarGANVC/2000.gen.npz \
    --output_prefix out_files_2/stargan_bigram/${tspk}/out \
    --mode ML \
    --flat_f0 \
    --flat_ap \
    --size ${size} \
    --unique

done

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
    --output_prefix out_files_2/stargan_lstm/${tspk}/out \
    --mode ML \
    --flat_f0 \
    --flat_ap \
    --size ${size} \
    --unique

done
