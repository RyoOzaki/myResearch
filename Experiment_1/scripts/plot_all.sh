#!/bin/bash

# sh plot.sh -r sgvc_all_speaker
result_dir="segmentation_result"

for result_name in `ls ${result_dir}`;
do
  sh scripts/plot.sh -r ${result_dir} -n ${result_name} &
done
