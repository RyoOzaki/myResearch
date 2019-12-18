#!/bin/bash

#==============================================
# normalize
# output
#   feature/norm_all_speaker.npz (normalize mcep_all_speaker_20msec.npz)
#   feature/norm_spkind_all_speaker.npz (normalize mcep_all_speaker_20msec.npz) # speaker individual normalization
#   feature/norm_speaker_H.npz (normalize mcep_speaker_H.npz)
#   feature/norm_speaker_K.npz (normalize mcep_speaker_K.npz)
#   feature/norm_speaker_M.npz (normalize mcep_speaker_M.npz)
#   feature/norm_speaker_N.npz (normalize mcep_speaker_N.npz)
python src/Normalize/gaussian_normalize.py --source_file feature/mcep_all_speaker_20msec.npz --output_file feature/norm_all_speaker.npz
python src/Normalize/gaussian_normalize.py --source_file feature/mcep_all_speaker_20msec.npz --output_file feature/norm_spkind_all_speaker.npz --speaker_id feature/speaker.npz
python src/Normalize/gaussian_normalize.py --source_file feature/mcep_speaker_H.npz --output_file feature/norm_speaker_H.npz
python src/Normalize/gaussian_normalize.py --source_file feature/mcep_speaker_K.npz --output_file feature/norm_speaker_K.npz
python src/Normalize/gaussian_normalize.py --source_file feature/mcep_speaker_M.npz --output_file feature/norm_speaker_M.npz
python src/Normalize/gaussian_normalize.py --source_file feature/mcep_speaker_N.npz --output_file feature/norm_speaker_N.npz
