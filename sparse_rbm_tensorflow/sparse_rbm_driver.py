import cPickle as pickle
import numpy as np
import sys
import tensorflow as tf
import math
import time
import random

from sparse_rbm import SparseRBMClass
from sparse_row_matrix import CompressedSparseRow
from sparse_rbm_params import SparseRBMParams

#
# This is a driver for the SparseRBMClass, which implements collaborative
# filtering via a RestrictedBoltzmannMachine.
#
# For each run through the entire dataset, this driver will evaluate
# the RBM on the provided test set and output a variety of information
# for debugging, including the RMSE. 
#
# As RBM parameters are often modified over the course of the training,
# this driver uses a SparseRBMParams input, which tracks the learning
# rate, number of Gibbs samples, momentum, and weight penalty for
# each epoch. See sparse_rbm_params for more information.
#

def evaluate_test_set(rbm, training_set, test_set, batch_size, num_hidden):
    squared_error = 0
    abs_error = 0
    predicted_ratings_list = []
    hidden_unit_activations = np.zeros([1, num_hidden])

    num_test_users = test_set.num_rows
    test_row_extents = test_set.row_extents_array
    test_indices = test_set.indexes_array
    test_ratings = test_set.ratings_array

    # Walk through the rows of the test set in batches
    for offset in range(0, num_test_users, batch_size):

        # If we are reaching the end of the array, truncate the input
        if ((offset + batch_size) > num_test_users):
            # At the end of the array, so need to trim the indices
            start = offset
            end = num_test_users
        else:
            # Not at the end of the array, do full batch
            start = offset
            end = offset + batch_size

        # Take the row indices for this batch and convert them to user indices
        test_user_indices = test_set.rowidx_2_useridx[start:end]

        # Take the user indices and use them to get the corresponding
        # row indices in the training set.
        testuser2trainingrow = lambda x: training_set.useridx_2_rowidx[x]
        train_row_indices = map(testuser2trainingrow, test_user_indices)

        # Send the training indices into the validation function
        # This returns the ratings for all items for those rows
        # It also returns the hidden unit information for debugging.
        visible_out = rbm.run_validation(end-start, train_row_indices)
        visible_array = visible_out[0]
        hidden_array = visible_out[1]

        # Aggregate the hidden units. It is useful to have information about
        # which hidden units are activating and whether there is any
        # variety in how frequently they are activating.
        for index in range(end-start):
            hidden_unit_activations += hidden_array[index]
            
        # Walk through each test set row
        for i in range(end-start):
            row_extent = test_row_extents[start+i]
            row_start = row_extent[0]
            row_size = row_extent[1]
            item_indices = test_indices[row_start:row_start+row_size]
            item_ratings = test_ratings[row_start:row_start+row_size]
            predicted_ratings = visible_array[i]
            user_index = test_user_indices[i]

            # For each item for the test user, calculate the predicted
            # rating and the corresponding error
            for j in range(len(item_indices)):
                item_index = item_indices[j]

                # Neither user nor item in training set
                if (item_index not in training_set.item_averages):
                    predicted_rating = total_sum / total_count
                elif (user_index not in training_set.user_dict):
                    # Look up item
                    item_entry = training_set.item_averages[item_index]
                    cur_sum = item_entry[0]
                    count = item_entry[1]
                    predicted_rating = cur_sum / count
                else:
                    # Both user and item are in training set
                    predicted_rating = predicted_ratings[item_index]
                    
                actual_rating = item_ratings[j]
                squared_error += pow(predicted_rating - actual_rating, 2)
                abs_error += abs(predicted_rating - actual_rating)
                predicted_ratings_list.append([user_index, item_index, predicted_rating, actual_rating])

    rmse = math.sqrt(squared_error / len(test_ratings))
    mabserr = abs_error / len(test_ratings)
    hidden_activations = hidden_unit_activations / num_test_users

    return rmse, mabserr, predicted_ratings_list, hidden_activations

def main():
    if len(sys.argv) != 5:
        print "Usage: {0} training_file test_file param_pickle output_dir".format(sys.argv[0])
        exit(1)

    training_filename = sys.argv[1]
    test_filename = sys.argv[2]
    param_pickle_filename = sys.argv[3]
    output_dir = sys.argv[4]

    training_set = CompressedSparseRow(training_filename)
    test_set = CompressedSparseRow(test_filename)

    param_pickle_f = open(param_pickle_filename, 'rb')
    param_settings = pickle.load(param_pickle_f)
    param_pickle_f.close()

    # Logging files
    settings_file = open(output_dir + "/settings.txt", "w")
    rmse_file = open(output_dir + "/rmse.txt", "w")
    mabserr_file = open(output_dir + "/mabserr.txt", "w")
    timings_file = open(output_dir + "/timings.txt", "w")

    num_users = training_set.num_rows
    num_test_users = test_set.num_rows
    
    batch_size = 100
    num_buckets = 5
    num_visible = training_set.max_item_idx + 1
    num_epochs = param_settings.num_epochs
    num_hidden = param_settings.num_hidden
    
    rbm = SparseRBMClass(num_hidden, num_buckets, batch_size, training_set, sparsity_mix=0.02)

    settings_file.write("Parameter information:\n")
    settings_file.write("num_hidden: {}\n".format(num_hidden))
    settings_file.write("num_epochs: {}\n".format(num_epochs))
    settings_file.write("batch size: {}\n".format(batch_size))
    
    settings_file.write("Training set information:\n")
    settings_file.write("train filename: {}\n".format(training_filename))
    settings_file.write(str(training_set))

    settings_file.write("Test set information:\n")
    settings_file.write("test filename: {}\n".format(test_filename))
    settings_file.write(str(test_set))

    # We shuffle the training set row indices and iterate over those.
    # This prevents any effects due to the ordering of the users.
    training_users_range = range(num_users)
    random.shuffle(training_users_range)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_num = epoch+1
        alpha, k, momentum, weight_penalty = param_settings.get_epoch_params(epoch_num)
        
        settings_file.write("epoch {}: k={}, alpha={}, momentum={}, weight_penalty: {}\n".format(epoch_num, k, alpha, momentum, weight_penalty))
        
        for offset in range(0, num_users, batch_size):
            
            if ((offset + batch_size) > num_users):
                start = offset
                end = num_users
            else:
                start = offset
                end = offset + batch_size
            
            training_indices = training_users_range[start:end]
            
            rbm.run_one_iteration(k, alpha, momentum, weight_penalty,
                                  end-start, training_indices)

        print "done with training epoch {}".format(epoch_num)

        # Evaluate the error on the test set
        rmse, mabserr, predicted_ratings_list, hidden_activations = \
            evaluate_test_set(rbm, training_set, test_set, batch_size, num_hidden)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        #print hidden_activations
        
        timings_file.write("epoch: {0} time: {1}\n".format(epoch_num, epoch_duration))
        rmse_file.write("epoch: {0} rmse: {1}\n".format(epoch_num, rmse))
        mabserr_file.write("epoch: {0} mabserr: {1}\n".format(epoch_num, mabserr))

        epoch_information = rbm.get_internal_state()

        epoch_info_filename = output_dir + "/epoch_{}.pkl".format(epoch_num)
        epoch_info_file = open(epoch_info_filename, "w")
        pickle.dump(epoch_information, epoch_info_file)
        epoch_info_file.close()

        if (epoch_num == num_epochs):
            predict_filename = output_dir + "/epoch_{}_predict.pkl".format(epoch_num)
            predict_file = open(predict_filename, "w")
            pickle.dump(predicted_ratings_list, predict_file)
            predict_file.close()
        
        print "epoch: {0} RMSE: {1} MAE: {2}".format(epoch_num, rmse, mabserr)

    settings_file.close()
    timings_file.close()
    rmse_file.close()
    mabserr_file.close()

if __name__=="__main__":
    main()
