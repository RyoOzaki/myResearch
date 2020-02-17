#!/bin/bash

# sh plot.sh -n sgvc_all_speaker_20msec -> segmentation_result/sgvc_all_speaker_20msec
result_dir="segmentation_result"
figure_dir="segmentation_result_figure"

mkdir -p ${figure_dir}

while getopts r:n: OPT
do
  case $OPT in
    "r" ) result_dir="${OPTARG}" ;;
    "n" ) result_name="${OPTARG}" ;;
  esac
done

dir_names=`ls ${result_dir}/${result_name} | grep "^[0-9]*$" -o`

for dname in ${dir_names};
do
  echo "Processing ${result_name}/${dname}"
  python src/Evaluate_summary/plot_segmentation_results.py \
    --model ${result_dir}/${result_name}/hypparams/model.config \
    --result_dir ${result_dir}/${result_name}/${dname}/results/ \
    --phn feature/phn_all_speaker_20msec.npz \
    --wrd feature/wrd_all_speaker_20msec.npz \
    --Ft_phn feature/Ft_phn_all_speaker_20msec.npz \
    --Ft_wrd feature/Ft_wrd_all_speaker_20msec.npz \
    --figure_dir ${figure_dir}/${result_name}/${dname}/
    cp -r ${result_dir}/${result_name}/${dname}/figures/* ${figure_dir}/${result_name}/${dname}/
done

cp -r ${result_dir}/${result_name}/figures ${figure_dir}/${result_name}/

echo "Finished ${result_name}!!"
