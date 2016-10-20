import sys
import string
import cPickle as pickle
import random
import math
import numpy as np

def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) != 8:
        print "Usage: {}: training_file test_file m_matrix u_matrix num_factors predictions_out rmse_out".format(argv[0])
        return 2

    train_in = argv[1]
    test_in = argv[2]
    user_matrix_in = argv[3]
    item_matrix_in = argv[4]
    num_factors = int(argv[5])
    predictions_out = argv[6]
    rmse_out = argv[7]

    f = open(train_in, 'r')

    item_dict = {}
    total_sum = 0
    total_count = 0
    max_user_index = 0
    max_item_index = 0
    
    for line in f:
        fields = string.split(line, '\t')
        user_index = int(fields[0])
        item_index = int(fields[1])
        rating = float(fields[2])

        if (user_index > max_user_index):
            max_user_index = user_index

        if (item_index > max_item_index):
            max_item_index = item_index
        
        total_sum += rating
        total_count += 1
        if (item_index in item_dict):
            item_entry = item_dict[item_index]
            cur_sum = item_entry[0]
            count = item_entry[1]
            item_dict[item_index] = [cur_sum + rating, count+1]
        else:
            item_dict[item_index] = [rating, 1]
    
    f.close()
    
    # Go through the test file to see if any user index or item index
    # exceeds what we see in the training set. This establishes the
    # size of the arrays needed.
    
    f = open(test_in, 'r')
    
    for line in f:
        fields = string.split(line, '\t')
        user_index = int(fields[0])
        item_index = int(fields[1])

        if (user_index > max_user_index):
            max_user_index = user_index

        if (item_index > max_item_index):
            max_item_index = item_index
    
    f.close()
    
    item_file = open(item_matrix_in, "rb")
    user_file = open(user_matrix_in, "rb")

    num_items = max_item_index + 1
    num_users = max_user_index + 1
    
    item_mat = np.zeros([num_items, num_factors])
    user_mat = np.zeros([num_users, num_factors])

    # Walk through the item CSV file
    for line in item_file:
        if len(line) == 1:
            continue
        
        split_line = string.split(line, "\t")
        item_index = int(split_line[0])
        item_factors_str = string.strip(split_line[1])
        item_factors = map(float, string.split(item_factors_str, ","))
        assert(len(item_factors) == num_factors)
        item_mat[item_index] = item_factors

    for line in user_file:
        if len(line) == 1:
            continue
        
        split_line = string.split(line, "\t")
        user_index = int(split_line[0])
        user_factors_str = string.strip(split_line[1])
        user_factors = map(float, string.split(user_factors_str, ","))
        assert(len(user_factors) == num_factors)
        user_mat[user_index] = user_factors

    # Shape is (num_items, num_users)
    # predicted_ratings = np.dot(m_mat, u_mat.T)

    testset_f = open(test_in, 'r')
    predict_g = open(predictions_out, "w")

    sqerr = 0
    abserr = 0
    mahout_sqerr = 0
    mahout_abserr = 0
    overall_average = total_sum/ total_count
    total_predictions = 0
    mahout_predictions = 0

    for line in testset_f:
        fields = string.split(line, '\t')
        user_index = int(fields[0])
        item_index = int(fields[1])
        actual_rating = float(fields[2])

        user_vector = user_mat[user_index]
        item_vector = item_mat[item_index]
        predicted_rating = np.dot(user_vector, item_vector)

        used_average_prediction = False
        if predicted_rating == 0.0:
            used_average_prediction = True
            # Use average rating for the prediction
            if (item_index in item_dict):
                item_entry = item_dict[item_index]
                cur_sum = item_entry[0]
                count = item_entry[1]
                predicted_rating = cur_sum / count
            else:
                predicted_rating = total_sum / total_count
        
        error = actual_rating - predicted_rating
        sqerr += pow(error, 2)
        abserr += abs(error)
        total_predictions += 1
        
        if (used_average_prediction):
            predict_g.write("{} {} {} ***\n".format(user_index, item_index, predicted_rating))
        else:
            mahout_sqerr += pow(error, 2)
            mahout_abserr += abs(error)
            mahout_predictions += 1
            predict_g.write("{} {} {}\n".format(user_index, item_index, predicted_rating))

    normal_rmse = math.sqrt(sqerr / total_predictions)
    mahout_rmse = math.sqrt(mahout_sqerr / mahout_predictions)

    f.close()
    predict_g.close()

    
    rmse_f = open(rmse_out, "w")
    
    rmse_f.write("Normal RMSE: {} Mahout RMSE: {}\n".format(normal_rmse, mahout_rmse))
    rmse_f.close()
    
if __name__=="__main__":
    sys.exit(main())
