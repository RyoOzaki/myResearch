#!/bin/bash

#==============================================
PCA_COMPONENTS="5"
# compress mcep using PCA
#   feature/pca_all_speaker.npz (compress mcep_all_speaker_20msec.npz)
#   feature/pca_speaker_H.npz (compress mcep_speaker_H.npz)
#   feature/pca_speaker_K.npz (compress mcep_speaker_K.npz)
#   feature/pca_speaker_M.npz (compress mcep_speaker_M.npz)
#   feature/pca_speaker_N.npz (compress mcep_speaker_N.npz)
python src/Compress/PCA_compress.py --source_file feature/mcep_all_speaker_20msec.npz --output_file feature/pca_all_speaker.npz --n_components ${PCA_COMPONENTS}
python src/Compress/PCA_compress.py --source_file feature/mcep_speaker_H.npz --output_file feature/pca_speaker_H.npz --n_components ${PCA_COMPONENTS}
python src/Compress/PCA_compress.py --source_file feature/mcep_speaker_K.npz --output_file feature/pca_speaker_K.npz --n_components ${PCA_COMPONENTS}
python src/Compress/PCA_compress.py --source_file feature/mcep_speaker_M.npz --output_file feature/pca_speaker_M.npz --n_components ${PCA_COMPONENTS}
python src/Compress/PCA_compress.py --source_file feature/mcep_speaker_N.npz --output_file feature/pca_speaker_N.npz --n_components ${PCA_COMPONENTS}

#==============================================
DSAE_STRUCTURE="36 18 9 5"
DSAE_PBHL_STRUCTURE="4 3"

# compress mcep using DSAE and DSAE-PBHL
#   feature/dsae_all_speaker.npz (compress mcep_all_speaker_20msec.npz)
#   feature/dsae_pbhl_all_speaker.npz (compress mcep_all_speaker_20msec.npz and speaker.npz)
#   feature/dsae_pbhl_v2_all_speaker.npz (compress mcep_all_speaker_20msec.npz and speaker.npz)
#   feature/dsae_speaker_H.npz (compress mcep_speaker_H.npz)
#   feature/dsae_speaker_K.npz (compress mcep_speaker_K.npz)
#   feature/dsae_speaker_M.npz (compress mcep_speaker_M.npz)
#   feature/dsae_speaker_N.npz (compress mcep_speaker_N.npz)
python src/DSAE-PBHL/DSAE_train.py --train_data feature/mcep_all_speaker_20msec.npz --output_file feature/dsae_all_speaker.npz --structure ${DSAE_STRUCTURE}
python src/DSAE-PBHL/DSAE_PBHL_train.py --train_data feature/mcep_all_speaker_20msec.npz --speaker_id feature/speaker.npz  --output_file feature/dsae_pbhl_all_speaker.npz --structure ${DSAE_STRUCTURE} --pb_structure ${DSAE_PBHL_STRUCTURE}
python src/DSAE-PBHL/DSAE_PBHL_v2_train.py --train_data feature/mcep_all_speaker_20msec.npz --speaker_id feature/speaker.npz  --output_file feature/dsae_pbhl_v2_all_speaker.npz --structure ${DSAE_STRUCTURE} --pb_structure ${DSAE_PBHL_STRUCTURE}
python src/DSAE-PBHL/DSAE_train.py --train_data feature/mcep_speaker_H.npz --output_file feature/dsae_speaker_H.npz --structure ${DSAE_STRUCTURE}
python src/DSAE-PBHL/DSAE_train.py --train_data feature/mcep_speaker_K.npz --output_file feature/dsae_speaker_K.npz --structure ${DSAE_STRUCTURE}
python src/DSAE-PBHL/DSAE_train.py --train_data feature/mcep_speaker_M.npz --output_file feature/dsae_speaker_M.npz --structure ${DSAE_STRUCTURE}
python src/DSAE-PBHL/DSAE_train.py --train_data feature/mcep_speaker_N.npz --output_file feature/dsae_speaker_N.npz --structure ${DSAE_STRUCTURE}

#==============================================

# compress mcep using StarGAN-VC
#   feature/sgvc_all_speaker.npz (compress norm_with_f0_spkind_all_speaker.npz)
#   feature/sgvc_new_all_speaker.npz (compress norm_with_f0_spkind_all_speaker.npz)
python src/StarGAN-VC/train_stargan-vc.py --train_data feature/norm_with_f0_spkind_all_speaker.npz --speaker_id feature/speaker.npz --output_file feature/sgvc_all_speaker.npz --batchsize 8 --gpu 1
python src/StarGAN-VC/train_stargan-vc_new.py --train_data feature/norm_with_f0_spkind_all_speaker.npz --speaker_id feature/speaker.npz --output_file feature/sgvc_new_all_speaker.npz --batchsize 8 --gpu 1
