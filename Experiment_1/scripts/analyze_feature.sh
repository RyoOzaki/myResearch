#!/bin/bash

feature_dir="feature"
output_dir="feature_figure"

mkdir -p ${output_dir}

for file in `ls ${feature_dir} | grep ".*20msec.*npz$" | grep -v "phn" | grep -v "wrd"`;
do
  echo ${file}

  python src/Feature_analysis/analize.py \
    --phn feature/phn_all_speaker_20msec.npz \
    --speaker_id feature/speaker.npz \
    --source ${feature_dir}/${file} \
    --with_pca \
    --n_components 2 \
    --mode phn \
    --savefig ${output_dir}/${file}_phn.png

  python src/Feature_analysis/analize.py \
    --phn feature/phn_all_speaker_20msec.npz \
    --speaker_id feature/speaker.npz \
    --source ${feature_dir}/${file} \
    --with_pca \
    --n_components 2 \
    --mode spk \
    --savefig ${output_dir}/${file}_spk.png
done
