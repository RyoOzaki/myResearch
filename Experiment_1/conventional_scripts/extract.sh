#!/bin/bash

#==============================================
# extract mfcc, phn, wrd, speaker_id by multispeaker_AIOI data
# output
#   feature_conv/mfcc_all_speaker_25_10msec.npz (renamed from mfcc.npz)
#   feature_conv/dmfcc_all_speaker_25_10msec.npz (renamed from dmfcc.npz)
#   feature_conv/ddmfcc_all_speaker_25_10msec.npz (renamed from ddmfcc.npz)
#   feature_conv/phn_all_speaker_25_10msec.npz
#   feature_conv/wrd_all_speaker_25_10msec.npz
#   feature_conv/Ft_phn_all_speaker_25_10msec.npz
#   feature_conv/Ft_wrd_all_speaker_25_10msec.npz
#   feature_conv/speaker.npz
python src/Extractor/extract.py \
  --source_dir dataset/ \
  --feature_type mfcc \
  --winlen 25E-3 \
  --winstep 10E-3 \
  --label_format time \
  --phn_label_extension lab \
  --wrd_label_extension lab2 \
  --sil_label s

python src/Extractor/collect2npz.py \
  --source_dir dataset/ \
  --output_dir feature_conv/ \
  --collect_extensions mfcc dmfcc ddmfcc phn wrd Ft_phn Ft_wrd \
  --with_speaker_id \
  --speaker_dir_layer 1

mv feature_conv/mfcc.npz feature_conv/mfcc_all_speaker_25_10msec.npz
mv feature_conv/dmfcc.npz feature_conv/dmfcc_all_speaker_25_10msec.npz
mv feature_conv/ddmfcc.npz feature_conv/ddmfcc_all_speaker_25_10msec.npz
mv feature_conv/wrd.npz feature_conv/wrd_all_speaker_25_10msec.npz
mv feature_conv/phn.npz feature_conv/phn_all_speaker_25_10msec.npz
mv feature_conv/Ft_wrd.npz feature_conv/Ft_wrd_all_speaker_25_10msec.npz
mv feature_conv/Ft_phn.npz feature_conv/Ft_phn_all_speaker_25_10msec.npz

#==============================================
# clean dataset dir
python src/Extractor/clean.py \
  --source_dir dataset/ \
  --remove_extensions mfcc dmfcc ddmfcc wrd Ft_wrd phn Ft_phn

#==============================================

python src/Extractor/concatenate.py \
  --sources \
    feature_conv/mfcc_all_speaker_25_10msec.npz \
    feature_conv/dmfcc_all_speaker_25_10msec.npz \
    feature_conv/ddmfcc_all_speaker_25_10msec.npz \
  --output feature_conv/concat_mfcc_all_speaker_25_10msec.npz

#==============================================
# pickup each speaker
# output
#   feature_conv/mfcc_speaker_H_25_10msec.npz (pickup by mfcc_all_speaker_25_10msec.npz)
#   feature_conv/mfcc_speaker_K_25_10msec.npz (pickup by mfcc_all_speaker_25_10msec.npz)
#   feature_conv/mfcc_speaker_M_25_10msec.npz (pickup by mfcc_all_speaker_25_10msec.npz)
#   feature_conv/mfcc_speaker_N_25_10msec.npz (pickup by mfcc_all_speaker_25_10msec.npz)

python src/Extractor/pickup.py \
  --source_file feature_conv/mfcc_all_speaker_25_10msec.npz \
  --output_file feature_conv/mfcc_speaker_H_25_10msec.npz \
  --keyword speaker_H

python src/Extractor/pickup.py \
  --source_file feature_conv/mfcc_all_speaker_25_10msec.npz \
  --output_file feature_conv/mfcc_speaker_K_25_10msec.npz \
  --keyword speaker_K

python src/Extractor/pickup.py \
  --source_file feature_conv/mfcc_all_speaker_25_10msec.npz \
  --output_file feature_conv/mfcc_speaker_M_25_10msec.npz \
  --keyword speaker_M

python src/Extractor/pickup.py \
  --source_file feature_conv/mfcc_all_speaker_25_10msec.npz \
  --output_file feature_conv/mfcc_speaker_N_25_10msec.npz \
  --keyword speaker_N
