#!/bin/bash

#==============================================
# Volume normalization
python src/Volume_normalizer/normalize.py \
  --source_dir dataset/ \
  --max_amplitude 30000 \
  --replace

#==============================================
# extract mcep, phn, wrd, speaker_id by multispeaker_AIOI data
# output
#   feature/mcep_all_speaker_20msec.npz (renamed from mcep.npz)
#   feature/phn_all_speaker_20msec.npz
#   feature/wrd_all_speaker_20msec.npz
#   feature/Ft_phn_all_speaker_20msec.npz
#   feature/Ft_wrd_all_speaker_20msec.npz
#   feature/speaker.npz
python src/Extractor/extract_world.py \
  --source_dir dataset/ \
  --feature_type mcep \
  --label_format time \
  --phn_label_extension lab \
  --wrd_label_extension lab2 \
  --sil_label s \
  --frame_period 20E-3 \
  --nfilt 36 \
  --fftsize 2048

python src/Extractor/extract.py \
  --source_dir dataset/ \
  --feature_type mfcc \
  --winlen 20E-3 \
  --winstep 20E-3 

python src/Extractor/collect2npz.py \
  --source_dir dataset/ \
  --output_dir feature/ \
  --collect_extensions mcep phn wrd Ft_phn Ft_wrd \
  --with_speaker_id \
  --speaker_dir_layer 1

mv feature/mcep.npz feature/mcep_all_speaker_20msec.npz
mv feature/wrd.npz feature/wrd_all_speaker_20msec.npz
mv feature/phn.npz feature/phn_all_speaker_20msec.npz
mv feature/Ft_wrd.npz feature/Ft_wrd_all_speaker_20msec.npz
mv feature/Ft_phn.npz feature/Ft_phn_all_speaker_20msec.npz

#==============================================
# extract mcep by multispeaker_AIOI data
# output
#   feature/mcep_all_speaker_5msec.npz (renamed from mcep.npz)
#   feature/f0_all_speaker_5msec.npz (renamed from f0.npz)
#   feature/ap_all_speaker_5msec.npz (renamed from ap.npz)
python src/Extractor/extract_world.py \
  --source_dir dataset/ \
  --feature_type mcep f0 ap \
  --frame_period 5E-3 \
  --fftsize 2048 \
  --nfilt 36

python src/Extractor/collect2npz.py \
  --source_dir dataset/ \
  --output_dir feature/ \
  --collect_extensions mcep f0 ap \
  --with_speaker_id \
  --speaker_dir_layer 1

mv feature/mcep.npz feature/mcep_all_speaker_5msec.npz
mv feature/f0.npz feature/f0_all_speaker_5msec.npz
mv feature/ap.npz feature/ap_all_speaker_5msec.npz

#==============================================
# pickup each speaker
# output
#   feature/mcep_speaker_H_20msec.npz (pickup by mcep_all_speaker_20msec.npz)
#   feature/mcep_speaker_K_20msec.npz (pickup by mcep_all_speaker_20msec.npz)
#   feature/mcep_speaker_M_20msec.npz (pickup by mcep_all_speaker_20msec.npz)
#   feature/mcep_speaker_N_20msec.npz (pickup by mcep_all_speaker_20msec.npz)
python src/Extractor/pickup.py \
  --source_file feature/mcep_all_speaker_20msec.npz \
  --output_file feature/mcep_speaker_H_20msec.npz \
  --keyword speaker_H

python src/Extractor/pickup.py \
  --source_file feature/mcep_all_speaker_20msec.npz \
  --output_file feature/mcep_speaker_K_20msec.npz \
  --keyword speaker_K

python src/Extractor/pickup.py \
  --source_file feature/mcep_all_speaker_20msec.npz \
  --output_file feature/mcep_speaker_M_20msec.npz \
  --keyword speaker_M

python src/Extractor/pickup.py \
  --source_file feature/mcep_all_speaker_20msec.npz \
  --output_file feature/mcep_speaker_N_20msec.npz \
  --keyword speaker_N

#==============================================
# clean dataset dir
python src/Extractor/clean.py \
  --source_dir dataset/ \
  --remove_extensions mfcc dmfcc ddmfcc mcep f0 ap wrd Ft_wrd phn Ft_phn
