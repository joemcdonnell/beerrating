#!/bin/bash

# When computing the RMSE, Mahout skips data points in the test set where
# there is not enough data to make a prediction.
#
# This script takes the output from a Mahout run
# (see mahout_process_dataset.sh) and computes the RMSE with the
# unknown values replaced with the training set average for the
# item (if item is seen) or the overall average otherwise.
#
# The input for this script is the same parameter file used for
# mahout_process_dataset.sh. See mahout_process_dataset.sh for expected
# parameter values.
#

. ${1}

rm ${output_dir}/final_output/user_mat.csv
rm ${output_dir}/final_output/item_mat.csv

# Dump the Mahout generated user matrix as a CSV file
mahout vectordump \
       --input ${output_dir}/final_output/U/ \
       --output ${output_dir}/final_output/user_mat.csv \
       --csv ${output_dir}/final_output/user_mat.csv \
       -p whatever

# Dump the Mahout generated item matrix as a CSV file
mahout vectordump \
       --input ${output_dir}/final_output/M/ \
       --output ${output_dir}/final_output/item_mat.csv \
       --csv ${output_dir}/final_output/item_mat.csv \
       -p whatever

python calculate_corrected_rmse.py \
       ${training_set} \
       ${test_set} \
       ${output_dir}/final_output/user_mat.csv \
       ${output_dir}/final_output/item_mat.csv \
       ${num_features} \
       ${output_dir}/test_set_predictions.txt \
       ${output_dir}/corrected_rmse.txt
