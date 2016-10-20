#!/bin/bash

# This script runs Mahout on a dataset and then generates the RMSE values
# both for the final result and the intermediate steps.

# Input comes in the form of a parameter file that defines the following
# parameters (in bash style). All parameters need to be defined.
#
# Input/Output parameters:
# training_set
# test_set
# output_directory
#
# Mahout tuning parameters:
# lambda
# num_iterations
# num_features
#
# Mahout runtime parameters:
# num_threads
#
# Example param file (beerrating50_params.txt):
#
# training_set=beerrating50_training.txt
# test_set=beerrating50_test.txt
# output_dir=beerrating50_lambda0.065_feature10
# lambda=0.065
# num_iterations=40
# num_features=10
# num_threads=12
#
# ./mahout_process_dataset.sh beerrating50_params.txt

. ${1}

# This call to mahout does the training. Mahout generates two matrices for
# each iteration: a user matrix and an item matrix. The final pair of matrices
# go in ${output_dir}/final_output. The intermediate versions go in
# ${output_dir}/intermediate_output.
#
# The datasets currently do not contain implicit data, so --implicitFeedback
# is set to false.
mahout parallelALS --input ${training_set} \
       --output ${output_dir}/final_output  \
       --lambda ${lambda}                   \
       --implicitFeedback false             \
       --numFeatures ${num_features}        \
       --numIterations ${num_iterations}    \
       --numThreadsPerSolver ${num_threads} \
       --tempDir ${output_dir}/intermediate_output

# Mahout uses a temp directory that it creates. If the temp directory already
# exists, Mahout will produce an error. So, delete the temp directory between
# Mahout runs
rm -rf temp

# Given a test set, Mahout will provide an RMSE value based on the matrices
# produced above. By producing the RMSE with the intermediate matrices as
# well, the RMSE can be graphed over the iterations.
#
# Description of the Mahout options:
# --input takes in the test set
# --output specifies a directory where mahout will put an rmse.txt file
# --userFeatures is the user matrix (the /U/ directory)
# --itemFeatures is the item matrix (the /M/ directory)
#
# Note: The --output option for evaluateFactorization is actually specifying
#       a directory where mahout will put an rmse.txt file.
#
# First, calculate the RMSE based on the final results

mahout evaluateFactorization \
       --input ${test_set} \
       --output ${output_dir}/rmse_final \
       --userFeatures ${output_dir}/final_output/U/ \
       --itemFeatures ${output_dir}/final_output/M/

num_intermediate=${num_iterations}
let num_intermediate-=2

# Go through each intermediate result and generate the RMSE

for i in `seq ${num_intermediate}`
do
  rm -rf temp
  mahout evaluateFactorization \
	 --input ${test_set} \
	 --output ${output_dir}/rmse_${i} \
	 --userFeatures ${output_dir}/intermediate_output/U-${i}/ \
	 --itemFeatures ${output_dir}/intermediate_output/M-${i}/
done

# Since all the rmse values are scattered across rmse.txt files in all the
# output directories that mahout used, combine them into a single rmses.txt
# file in the output directory. They are ordered from earlier iterations
# to later iterations with the final value last.

for i in `seq ${num_intermediate}`
do
    # The rmse.txt files do not contain a newline, so echo a newline
    # after catting each rmse.txt file.
    cat ${output_dir}/rmse_${i}/rmse.txt >> ${output_dir}/rmses.txt
    echo >> ${output_dir}/rmses.txt
done

cat ${output_dir}/rmse_final/rmse.txt >> ${output_dir}/rmses.txt
echo >> ${output_dir}/rmses.txt
