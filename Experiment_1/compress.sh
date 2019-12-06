#!/bin/bash

#==============================================
DSAE_STRUCTURE="36 20 10 5"
DSAE_PBHL_STRUCTURE="4 3"

# compress mcep using DSAE and DSAE-PBHL
#   feature/dsae_all_speaker.npz (compress mcep_all_speaker.npz)
#   feature/dsae_pbhl_all_speaker.npz (compress mcep_all_speaker.npz and speaker.npz)
#   feature/dsae_speaker_H.npz (compress mcep_speaker_H.npz)
#   feature/dsae_speaker_K.npz (compress mcep_speaker_K.npz)
#   feature/dsae_speaker_M.npz (compress mcep_speaker_M.npz)
#   feature/dsae_speaker_N.npz (compress mcep_speaker_N.npz)
python src/DSAE-PBHL/DSAE_train.py --train_data feature/mcep_all_speaker.npz --output_file feature/dsae_all_speaker.npz --structure ${DSAE_STRUCTURE}
python src/DSAE-PBHL/DSAE_PBHL_train.py --train_data feature/mcep_all_speaker.npz --speaker_id feature/speaker.npz  --output_file feature/dsae_pbhl_all_speaker.npz --structure ${DSAE_STRUCTURE} --pb_structure ${DSAE_PBHL_STRUCTURE}
python src/DSAE-PBHL/DSAE_train.py --train_data feature/mcep_speaker_H.npz --output_file feature/dsae_speaker_H.npz --structure ${DSAE_STRUCTURE}
python src/DSAE-PBHL/DSAE_train.py --train_data feature/mcep_speaker_K.npz --output_file feature/dsae_speaker_K.npz --structure ${DSAE_STRUCTURE}
python src/DSAE-PBHL/DSAE_train.py --train_data feature/mcep_speaker_M.npz --output_file feature/dsae_speaker_M.npz --structure ${DSAE_STRUCTURE}
python src/DSAE-PBHL/DSAE_train.py --train_data feature/mcep_speaker_N.npz --output_file feature/dsae_speaker_N.npz --structure ${DSAE_STRUCTURE}
