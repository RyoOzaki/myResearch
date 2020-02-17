#!/bin/bash

result_dir="segmentation_result"

for dirpath in `ls ${result_dir} | grep ".*_all_speaker_.*"`;
do
  for i in `seq 1 20`
  do
    i_str=$( printf '%02d' $i )
    echo "${result_dir}/${dirpath}/${i_str}"
    python src/NPB-DAA/summary.py \
      --phn_label ${result_dir}/${dirpath}/phn_all_speaker_20msec.npz \
      --wrd_label ${result_dir}/${dirpath}/wrd_all_speaker_20msec.npz \
      --speaker_id feature/speaker.npz \
      --model ${result_dir}/${dirpath}/hypparams/model.config \
      --results_dir ${result_dir}/${dirpath}/${i_str}/results \
      --figure_dir ${result_dir}/${dirpath}/${i_str}/figures \
      --summary_dir ${result_dir}/${dirpath}/${i_str}/summary_files
  done
  python src/NPB-DAA/summary_summary.py \
    --result_dir ${result_dir}/${dirpath} \
    --figure_dir ${result_dir}/${dirpath}/figures \
    --summary_dir ${result_dir}/${dirpath}/summary_files
done

dirs="segmentation_result/"

for dirpath in `ls ${result_dir} | grep ".*speaker_[HKMN]_.*"`;
do
  for i in `seq 1 20`
  do
    i_str=$( printf '%02d' $i )
    echo "${result_dir}/${dirpath}/${i_str}"
    python src/NPB-DAA/summary.py \
      --phn_label ${result_dir}/${dirpath}/phn_all_speaker_20msec.npz \
      --wrd_label ${result_dir}/${dirpath}/wrd_all_speaker_20msec.npz \
      --model ${result_dir}/${dirpath}/hypparams/model.config \
      --results_dir ${result_dir}/${dirpath}/${i_str}/results \
      --figure_dir ${result_dir}/${dirpath}/${i_str}/figures \
      --summary_dir ${result_dir}/${dirpath}/${i_str}/summary_files
  done
  python src/NPB-DAA/summary_summary.py \
    --result_dir ${result_dir}/${dirpath} \
    --figure_dir ${result_dir}/${dirpath}/figures \
    --summary_dir ${result_dir}/${dirpath}/summary_files
done
