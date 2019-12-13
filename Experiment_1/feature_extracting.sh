#!/bin/bash

#==============================================
# extract mcep, phn, wrd, speaker_id by multispeaker_AIOI data
# output
#   feature/mcep_all_speaker.npz (renamed from mcep.npz)
#   feature/phn.npz
#   feature/wrd.npz
#   feature/speaker.npz
python src/Extractor/extract_world.py --source_dir dataset/ --feature_type mcep --label_format time --phn_label_extension lab --wrd_label_extension lab2 --frame_period 20E-3 --nfilt 36
python src/Extractor/collect2npz.py --source_dir dataset/ --output_dir feature/ --collect_extensions mcep phn wrd --with_speaker_id --speaker_dir_layer 1
mv feature/mcep.npz feature/mcep_all_speaker.npz

#==============================================
# extract mcep by multispeaker_AIOI data
# output
#   feature/mcep_all_speaker_5msec.npz (renamed from mcep.npz)
python src/Extractor/extract_world.py --source_dir dataset/ --feature_type mcep --frame_period 5E-3 --nfilt 36
python src/Extractor/collect2npz.py --source_dir dataset/ --output_dir feature/ --collect_extensions mcep --with_speaker_id --speaker_dir_layer 1
mv feature/mcep.npz feature/mcep_all_speaker_5msec.npz

#==============================================
# pickup each speaker
# output
#   feature/mcep_speaker_H.npz (pickup by mcep_all_speaker.npz)
#   feature/mcep_speaker_K.npz (pickup by mcep_all_speaker.npz)
#   feature/mcep_speaker_M.npz (pickup by mcep_all_speaker.npz)
#   feature/mcep_speaker_N.npz (pickup by mcep_all_speaker.npz)
python src/Extractor/pickup.py --source_file feature/mcep_all_speaker.npz --output_file feature/mcep_speaker_H.npz --keyword speaker_H
python src/Extractor/pickup.py --source_file feature/mcep_all_speaker.npz --output_file feature/mcep_speaker_K.npz --keyword speaker_K
python src/Extractor/pickup.py --source_file feature/mcep_all_speaker.npz --output_file feature/mcep_speaker_M.npz --keyword speaker_M
python src/Extractor/pickup.py --source_file feature/mcep_all_speaker.npz --output_file feature/mcep_speaker_N.npz --keyword speaker_N

#==============================================
# clean dataset dir
python src/Extractor/clean.py --source_dir dataset/ --remove_extensions mcep wrd phn
