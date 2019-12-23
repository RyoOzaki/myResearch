#!/bin/bash

dirs="segmentation_result/*"

for dirpath in ${dirs};
do
  # echo ${dirpath}
  for i in `seq 1 20`
  do
    i_str=$( printf '%02d' $i )
    echo "${dirpath}/${i_str}"
    python src/NPB-DAA/summary.py \
      --phn_label ${dirpath}/phn_all_speaker_20msec.npz \
      --wrd_label ${dirpath}/wrd_all_speaker_20msec.npz \
      --model ${dirpath}/hypparams/model.config \
      --results_dir ${dirpath}/${i_str}/results \
      --figure_dir ${dirpath}/${i_str}/figures \
      --summary_dir ${dirpath}/${i_str}/summary_files
  done
  python src/NPB-DAA/summary_summary.py \
    --result_dir ${dirpath} \
    --figure_dir ${dirpath}/figures \
    --summary_dir ${dirpath}/summary_files
done
