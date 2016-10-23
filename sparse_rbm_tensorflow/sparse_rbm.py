import numpy as np
import tensorflow as tf
import math
from sparse_row_matrix import CompressedSparseRow

#
# This implements a Restricted Boltzmann Machine for use in collaborative
# filtering as described in Salakhutdinov et al 2007. It has softmax
# inputs with a number of buckets specified (currently set at 5).
#
# The sparse input is a CompressedSparseRow, implemented in
# sparse_row_matrix.py. The important point is that this format
# is organized around row indexes. Each row index corresponds to a user
# that has ratings of some number of items. For both 
#
# Due to the sparse input, the implementation is significantly different
# from a standard RBM. There are multiple approaches for implementing
# sparse matrix multiplication (which is the heart of an RBM). This
# implementation expands the input to non-sparse size and does a full
# matrix multiplication. This design trade-off makes the implementation
# expensive for very sparse inputs, but it allows for easy execution
# of the matrix multiplication on the GPU via Tensorflow. It also makes
# batching easy to implement.
#
# This RBM implementation also uses several improvements presented in
# Hinton's "A Practical Guide to Training Restricted Boltzmann Machines"
# Specifically, it implements momentum, a weight penalty, and batching.
#
# Usage:
# Creating the SparseRBMClass object will setup the callgraph and
# load the provided sparse dataset into TensorFlow.
#
# For each training iteration, the user needs to call the run_one_iteration
# method. This requires sending in a list of row indexes. This is the chunk
# of the training set that the RBM should train on. See sparse_rbm_driver.py
# for one way of running the training.
#
#

true_const = tf.constant(True, dtype=tf.bool)
false_const = tf.constant(False, dtype=tf.bool)

# This function takes a value in range 1.0 to 5.0
# and converts it to a 5 bucket representation
def bucketize_ratings(ratings):
    
    buckets = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    bucket_ratings = []
    for rating in ratings:
        
        this_rating = [0.0] * 5
        for i in range(len(buckets)):
            lower = buckets[i]
            upper = buckets[i+1]
            
            if (rating >= lower and rating <= upper):
                this_rating[i] = (upper - rating) / (upper - lower)
                this_rating[i+1] = 1 - this_rating[i]
                break
        
        bucket_ratings.extend(this_rating)
    
    return bucket_ratings
            
class SparseRBMClass:
    def __init__(self, num_hidden, num_buckets, max_batch_size, sparse_row_matrix, sparsity_mix=None):


        # Data input
        num_visible = sparse_row_matrix.max_item_idx + 1
        row_extents = sparse_row_matrix.row_extents_array
        ratings = sparse_row_matrix.ratings_array
        indexes = sparse_row_matrix.indexes_array
        
        # Initialization parameters
        # Total visible elements = num_visible * num_buckets
        # Total hidden elements = num_hidden
        self.num_visible = num_visible
        self.num_buckets = num_buckets
        self.num_hidden = num_hidden
        self.sparsity_mix = sparsity_mix
        self.max_batch_size = max_batch_size

        # Initialization data
        self.indexes = tf.Variable(tf.convert_to_tensor(indexes))
        self.row_extents = tf.Variable(tf.convert_to_tensor(row_extents))

        buckets = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        buckets_expanded = np.reshape(buckets, [self.num_buckets, 1])
        
        self.buckets_average = tf.Variable(tf.convert_to_tensor(buckets_expanded))
        # bucketize ratings (use 5 buckets for now)
        bucketized_ratings = bucketize_ratings(ratings)
        self.ratings = tf.Variable(tf.convert_to_tensor(bucketized_ratings))

        
        # Variables to train
        self.weights = tf.Variable(
            tf.random_uniform((self.num_visible * self.num_buckets, self.num_hidden),
                              -0.005, 0.005))
        self.hidden_bias = tf.Variable(tf.zeros((1, self.num_hidden)))
        self.visible_bias = tf.Variable(
            tf.random_uniform((1, self.num_visible * self.num_buckets), -0.005, 0.005))
        self.last_weight_update = tf.Variable(
            tf.zeros((self.num_visible * self.num_buckets, self.num_hidden)))
        self.last_visible_update = tf.Variable(
            tf.zeros((1, self.num_visible * self.num_buckets)))
        self.last_hidden_update = tf.Variable(
            tf.zeros((1, self.num_hidden)))

        # Input of the row indices for validation or a training iteration
        self.row_indices = tf.placeholder(tf.int64)
        
        # Tuning parameters
        self.learningrate = tf.placeholder(tf.float32)
        self.momentum = tf.placeholder(tf.float32)
        self.samples = tf.placeholder(tf.int32)
        self.weight_penalty = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32)

        # Build the actual execution graphs
        self.update_graph = None
        self.validation_graph = None

        self.build_update_graph()
        self.build_validation_graph()
        
        # The tensorflow session
        self.tf_session = None
        
        self.start_session()

    # Given a matrix of probabilities in [0,1], produce output matrix of
    # the same size with entries sampled uniformly using that probability.
    def binary_rand_sample(self, x):
        return tf.stop_gradient(tf.floor(tf.add(tf.random_uniform(tf.shape(x), 0, 1), x)))
    
    # visible is a b x v matrix.
    # weights is a v x h matrix.
    # hidden_bias is a 1 x h row vector
    # Output is b x h matrix
    def sample_hidden(self, visible, sample, data_mask):
        # Since we can have different inputs with different numbers of
        # items specified, we want to scale the hidden bias to account
        # for this. Specifically, multiply the bias by the number of items
        # specified for each vector.

        # Reduce the mask to get a count of the total number of items
        # specified. The mask has num_buckets values per item specified,
        # so multiply by num_buckets.
        collapse_mask = tf.stop_gradient(tf.div(tf.reduce_sum(data_mask, 1, keep_dims=True), self.num_buckets))
        hidden_bias_mat = tf.stop_gradient(tf.matmul(collapse_mask, self.hidden_bias))
        visible_inputs = tf.matmul(visible, self.weights)
        hidden_probs = tf.stop_gradient(tf.sigmoid(tf.add(visible_inputs, hidden_bias_mat)))

        # For batch_size != max_batch_size, need to mask out unused elements
        signed_collapse_mask = tf.sign(collapse_mask)
        tiled_mask_sign = tf.tile(signed_collapse_mask, [1, self.num_hidden])
        masked_hidden_probs = tf.mul(hidden_probs, tiled_mask_sign)
        
        return tf.cond(sample, lambda: self.binary_rand_sample(masked_hidden_probs), lambda: masked_hidden_probs)

    # Hidden can be a single vector or it can be a matrix (when batching)
    # hidden is a b x h matrix
    # weights is a v x h matrix (needs to be transposed)
    # visible_bias is a 1 x v row vector
    # Output is a b x v matrix
    def sample_visible(self, hidden, sample, data_mask):
        # We do transpose operations for efficiency
        tile_shape = tf.concat(0, [tf.expand_dims(self.max_batch_size, 0), [1]])

        # The visible bias applies to each of the visible inputs, so
        # tile it for each vector in the batch.
        visible_bias_mat = tf.stop_gradient(tf.tile(self.visible_bias, tile_shape))

        # Input to visible = h*W + visible_bias
        visible_inputs = tf.add(tf.matmul(hidden, self.weights, transpose_b=True), visible_bias_mat)

        # Do a softmax. Softmax applies to each row. Reshape so that each
        # individual rating is on its own row.
        visible_inputs_reshaped = tf.reshape(visible_inputs, [self.max_batch_size * self.num_visible, self.num_buckets])
        softmax_visible = tf.nn.softmax(visible_inputs_reshaped)
        
        # Convert back to normal shape
        visible_probs = tf.stop_gradient(tf.reshape(softmax_visible, [self.max_batch_size, self.num_visible * self.num_buckets]))

        # Mask out the elements that are not input ratings
        masked_visible_probs = tf.stop_gradient(tf.mul(data_mask, visible_probs))
        # If sample, then do a random sample. Otherwise, return the probs
        return tf.cond(sample, lambda: self.binary_rand_sample(masked_visible_probs), lambda: masked_visible_probs)

    def rbmGibbs(self, v, h, data_mask, count, visible_sample):
        # Hidden -> Visible
        v_prime = self.sample_visible(h, visible_sample, data_mask)

        # Visible -> Hidden
        # Do a random sample if this is not the last sample
        h_prime = self.sample_hidden(v_prime, tf.not_equal(count, self.samples-1), data_mask)
        return v_prime, h_prime, data_mask, count+1, visible_sample

    def less_than_samples(self, v, h, data_mask, count, visible_sample):
        return tf.less(count, self.samples)

    def expand_visible_vector(self, row_info):
        row_extent = tf.reshape(row_info, [2,1])
        row_begin = tf.gather(row_extent, 0)
        row_size = tf.gather(row_extent, 1)

        # Retrieve the indexes of the items for this row.
        # This produces a num_indices array
        sliced_indices = tf.expand_dims(tf.slice(self.indexes, begin=row_begin, size=row_size), 1)

        # Retrieve the ratings of the items for this row.
        #
        # Since each rating has multiple buckets, the indices and sizes
        # need to be multiplied by the number of buckets to cover the
        # appropriate elements.
        #
        # This produces a num_buckets * num indices array
        row_rating_begin = tf.mul(row_begin, self.num_buckets)
        row_rating_size = tf.mul(row_size, self.num_buckets)
        sliced_ratings = tf.slice(self.ratings, begin=row_rating_begin, size=row_rating_size);

        # At this point, to form the final visible input, we need to place
        # the ratings at the appropriate indices so that it forms
        # a num_buckets * num_visible array.
        #
        # To do that, we are going to expand the sliced_indices so that they
        # correspond to the correct indices for each of the values of
        # the sliced_ratings. 
        #
        # The appropriate index for an individual rating is:
        # [start_index, start_index + 1, ... start_index + num_buckets - 1]
        #
        # We will construct this in two parts. The first part is to generate
        # the part where start_index is correct and duplicated. The second
        # part is to generate the offsets (0, 1, ... num_buckets - 1). 
        #
        # Example:
        # Input = sliced_indices = [1, 7, 12, 17]
        #
        # First, multiply it by the num buckets (e.g. 3)
        # [3, 21, 36, 51]
        #
        # Then, tile it to duplicate the entries
        # [3,3,3, 21,21,21, 36,36,36, 51,51,51]
        tiled_sliced_indices = tf.tile(tf.mul(sliced_indices, self.num_buckets), [1, self.num_buckets])
        tile_size = tf.shape(sliced_indices)

        # Create a range up to num buckets for each sliced index
        # [0,1,2, 0,1,2, 0,1,2, 0,1,2]
        tiled_range = tf.tile(tf.expand_dims(tf.range(self.num_buckets), 0), tile_size)
        # The combination of these two things is the exact indices where
        # each rating (which has size num_buckets) should be stored
        # [3,4,5, 21,22,23, 36,37,38, 51,52,53]
        total_sliced_indices = tf.reshape(tf.add(tiled_sliced_indices, tiled_range), [-1])

        # Take each rating and copy it to the appropriate place in the
        # num_buckets * num_visible sized array.
        expanded_ratings = tf.expand_dims(tf.sparse_to_dense(total_sliced_indices, tf.constant([self.num_visible * self.num_buckets]), sliced_ratings, default_value=0.0), 0)
        # Now, flatten it into a 1 x (num_buckets * num_visible) array
        onehot_flattened = tf.reshape(expanded_ratings, [1, self.num_buckets * self.num_visible])
        
        return onehot_flattened

    def expand_visible_mask(self, row_info):
        row_extent = tf.reshape(row_info, [2,1])
        row_begin = tf.gather(row_extent, 0)
        row_size = tf.gather(row_extent, 1)

        # Retrieve the indexes of the items for this row.
        # This produces a num_indices array
        sliced_indices = tf.slice(self.indexes, begin=row_begin, size=row_size)

        # This function needs to produce a mask where all the visible elements
        # that have a rating are set to 1.0 and all the other elements
        # are zero.
        #
        # The final array is num_visible * num_buckets.
        #
        # First, we will create a num_visible sized array with the appropriate
        # elements set to 1.
        #
        # Then, we will tile this array and reshape it to be the right size.

        # Set appropriate indices to 1.0
        expanded_mask = tf.expand_dims(tf.sparse_to_dense(sliced_indices, tf.constant([self.num_visible]), 1.0, default_value=0.0), 1)

        # Tile it
        tiled_mask = tf.tile(expanded_mask, [1, self.num_buckets])

        # Reshape it
        flattened_mask = tf.reshape(tiled_mask, [1, self.num_buckets * self.num_visible])
        
        return flattened_mask
        
    def cond_expand_row(self, row_info):
        # If the row is empty, the expanded representation is all zero.
        # Otherwise, go through the expansion.
        return tf.cond(tf.equal(tf.size(row_info), 0), lambda: tf.zeros([1, self.num_visible*self.num_buckets]), lambda: self.expand_visible_vector(row_info))

    def cond_expand_row_mask(self, row_info):
        # If the row is empty, the mask is all zero. Otherwise, go through
        # the expansion.
        return tf.cond(tf.equal(tf.size(row_info), 0), lambda: tf.zeros([1, self.num_visible*self.num_buckets]), lambda: self.expand_visible_mask(row_info))

    def generate_visible_matrix(self):
        # Takes in an array of row indices
        row_index_array = tf.stop_gradient(tf.gather(self.row_extents, self.row_indices))

        # Explode the row array into a list of row indices
        row_list = tf.dynamic_partition(row_index_array, tf.range(0, self.batch_size), self.max_batch_size)

        # For each row index, expand it to produce the visible elements
        # This produces a list of expanded rows
        onehot_visible = map(self.cond_expand_row, row_list)

        # For each row index, expand it to produce a mask of the visible
        # elements that are filled in
        # This produces a list of expanded row masks
        onehot_mask = map(self.cond_expand_row_mask, row_list)

        # Collapase the lists to produce a single array for both the
        # expanded rows and the expanded row mask
        visible_start = tf.stop_gradient(tf.concat(0, onehot_visible))
        data_mask = tf.stop_gradient(tf.to_float(tf.concat(0, onehot_mask)))

        return (visible_start, data_mask)
    
    def build_update_graph(self):
        # Generate the visible matrix and the corresponding visible
        # mask matrix.
        (visible_start, data_mask) = self.generate_visible_matrix()

        # The number of Gibbs samples is configurable. It returns the
        # final visible and hidden vectors after the appropriate
        # number of iterations.
        count = tf.constant(0)
        hidden_start = self.sample_hidden(visible_start, true_const, data_mask)
        [visible_new, hidden_new, _, _, _] = tf.while_loop(
            self.less_than_samples, self.rbmGibbs,
            [visible_start, hidden_start, data_mask, count, true_const],
            parallel_iterations=1, back_prop=False)

        # A sparsity mix modifies the starting hidden value to push them towards
        # sparsity. The idea is that hidden units should be very selective
        # about activating.
        #
        # This is experimental.
        if (self.sparsity_mix):

            # Push the hidden values a little bit towards 0. 
            unmask_hidden_mod = tf.add(tf.mul(hidden_start, 1-self.sparsity_mix), tf.mul(tf.mul(tf.ones_like(hidden_start), 0.1), self.sparsity_mix))

            # need to mask hidden_mod?
            collapse_mask = tf.stop_gradient(tf.div(tf.reduce_sum(data_mask, 1, keep_dims=True), self.num_buckets))
            
            # For batch_size != max_batch_size, need to mask out unused elements
            signed_collapse_mask = tf.sign(collapse_mask)
            tiled_mask_sign = tf.tile(signed_collapse_mask, [1, self.num_hidden])
            hidden_mod = tf.mul(unmask_hidden_mod, tiled_mask_sign)
            hidden_start = hidden_mod

            
        # primary weight update term
        weight_update_1 = tf.stop_gradient(tf.mul(
            self.learningrate / tf.to_float(self.batch_size),
            tf.sub(
                tf.matmul(tf.transpose(visible_start), hidden_start),
                tf.matmul(tf.transpose(visible_new), hidden_new))))
        
        # momentum term
        weight_update_2 = tf.stop_gradient(tf.mul(self.momentum, self.last_weight_update))
        
        # weight penalty term
        weight_update_3 = tf.stop_gradient(tf.mul(self.learningrate,
                                 tf.mul(self.weight_penalty, self.weights)))
        
        # Total weight update
        weight_update = tf.add(weight_update_1,
                               tf.sub(weight_update_2, weight_update_3))
        
        # Hidden bias update
        # Normal term
        hidden_bias_update_1 = tf.stop_gradient(tf.mul(
            self.learningrate / tf.to_float(self.batch_size),
            tf.reduce_sum(tf.sub(hidden_start, hidden_new),
                          reduction_indices=0, keep_dims=True)))

        # Hidden bias momentum
        hidden_bias_update_2 = tf.mul(self.momentum, self.last_hidden_update)

        # Hidden bias weight penalty
        hidden_bias_update_3 = tf.mul(self.learningrate,
                                      tf.mul(self.weight_penalty, self.hidden_bias))

        # Cumulative hidden bias update
        hidden_bias_update = tf.sub(tf.add(hidden_bias_update_1, hidden_bias_update_2), hidden_bias_update_3)
        
        # Visible bias update
        # Normal term
        visible_bias_update_1 = tf.stop_gradient(tf.mul(
            self.learningrate / tf.to_float(self.batch_size),
            tf.reduce_sum(tf.sub(visible_start, visible_new),
                          reduction_indices=0, keep_dims=True)))
        
        # Visible bias momentum
        visible_bias_update_2 = tf.mul(self.momentum, self.last_visible_update)

        # Cumulative visible bias update
        visible_bias_update = tf.add(visible_bias_update_1,
                                     visible_bias_update_2)

        # Modify the weights, biases, etc.
        self.update_graph = [self.weights.assign_add(weight_update),
                             self.hidden_bias.assign_add(hidden_bias_update),
                             self.visible_bias.assign_add(visible_bias_update),
                             self.last_weight_update.assign(weight_update),
                             self.last_visible_update.assign(visible_bias_update),
                             self.last_hidden_update.assign(hidden_bias_update)]


    def build_validation_graph(self):
        (visible_start, actual_data_mask) = self.generate_visible_matrix()

        # We want to keep all of the values, because we are using this
        # to generate values for items not in the input. So, use a fake
        # mask with all 1.0's so that nothing gets masked.
        fake_data_mask = tf.ones_like(visible_start, dtype=tf.float32)
        
        #visible_start = self.visible_input
        hidden_start = self.sample_hidden(visible_start, false_const,
                                          actual_data_mask)
        count = tf.constant(0)
        
        [visible_new, hidden_new, _, _, _] = tf.while_loop(
            self.less_than_samples, self.rbmGibbs,
            [visible_start, hidden_start, fake_data_mask, count, false_const],
            parallel_iterations=1, back_prop=False)

        # Reshape so that each rating's buckets are a row
        visible_new_reshaped = tf.reshape(visible_new, [self.max_batch_size * self.num_visible, self.num_buckets])

        # Convert the buckets back to normal ratings by multiplying by
        # the bucket averages, then get it in the right shape.
        unbucket_visible = tf.matmul(visible_new_reshaped, self.buckets_average)
        unbucket_reshape = tf.reshape(unbucket_visible, [self.max_batch_size, self.num_visible])
        
        self.validation_graph = [unbucket_reshape, hidden_start]

    def start_session(self):
        self.tf_session = tf.Session()
        init = tf.initialize_all_variables()
        self.tf_session.run(init)
        
    def run_one_iteration(self, samples_in,
                          learningrate_in, momentum_in, weight_penalty_in,
                          batch_size_in, row_indices_in):
        self.tf_session.run(self.update_graph,
                            feed_dict={self.samples: samples_in,
                                       self.learningrate: learningrate_in,
                                       self.momentum: momentum_in,
                                       self.weight_penalty: weight_penalty_in,
                                       self.batch_size: batch_size_in,
                                       self.row_indices: row_indices_in})

    def run_validation(self, batch_size_in, row_indices_in):
        visible_out = \
            self.tf_session.run(self.validation_graph,
                                feed_dict={self.samples: 1,
                                           self.batch_size: batch_size_in,
                                           self.row_indices: row_indices_in})
        return visible_out

    def get_internal_state(self):
        weights = self.tf_session.run(self.weights)
        visible_bias = self.tf_session.run(self.visible_bias)
        hidden_bias = self.tf_session.run(self.hidden_bias)

        return [weights, visible_bias, hidden_bias]
