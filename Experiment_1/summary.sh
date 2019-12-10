#!/bin/bash

zip -r segmentation_result_backup.zip segmentation_result/
python src/Evaluate_summary/pickup_all_result.py --result_dir segmentation_result --output_dir segmentation_result_summary
python src/Evaluate_summary/show_results.py --result_dir segmentation_result_summary
