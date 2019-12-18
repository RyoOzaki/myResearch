#!/bin/bash

# sh plot.sh -r sgvc_all_speaker

rdirs=`ls segmentation_result`

for rdir in ${rdirs};
do
  sh plot.sh -r ${rdir} & 
done
