#!/bin/bash

zip -r segmentation_result_conv_backup.zip segmentation_result_conv/

python src/Evaluate_summary/pickup_all_result.py \
  --result_dir segmentation_result_conv \
  --output_dir segmentation_result_conv_summary

python src/Evaluate_summary/show_results.py \
  --result_dir segmentation_result_conv_summary
