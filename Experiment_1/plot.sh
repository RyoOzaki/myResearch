#!/bin/bash

# sh plot.sh -r sgvc_all_speaker

while getopts r: OPT
do
  case $OPT in
    "r" ) result_dir="${OPTARG}" ;;
  esac
done

dir_names=`ls segmentation_result/${result_dir} | grep "^[0-9]*$" -o`

for dname in ${dir_names};
do
  echo "Processing ${result_dir}/${dname}"
  python src/Evaluate_summary/plot_segmentation_results.py \
    --model segmentation_result/${result_dir}/hypparams/model.config \
    --result_dir segmentation_result/${result_dir}/${dname}/results/ \
    --phn feature/phn.npz \
    --wrd feature/wrd.npz \
    --Ft_phn feature/Ft_phn.npz \
    --Ft_wrd feature/Ft_wrd.npz \
    --figure_dir segmentation_result_figure/${result_dir}/${dname}/
done

cp -r segmentation_result/${result_dir}/figures segmentation_result_figure/${result_dir}/

echo "Finished ${result_dir}!!"
