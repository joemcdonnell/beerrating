import sys
import cPickle as pickle

#
# When training RBMs, certain parameters often change over time as the
# training progresses. For example, the number of Gibbs samples often
# starts at 1, but then increases as training gets closer to a solution.
# The learning rate can also be reduced as training progresses.
#
# This implements a basic parameter facility that allows the user
# to specify these parameters for each of the epochs. This allows
# drivers, such as sparse_rbm_driver.py, to fetch the parameter values
# for each epoch without anything hardcoded.
#
# TODO: make this able to load/save to a text format
#

def set_epoch_value(start_epoch, value, pair_array):
    # Is this start_epoch already present?
    index_to_delete = None
    for idx, pairs in enumerate(pair_array):
        if pairs[0] == start_epoch:
            index_to_delete = idx
            break
    
    if index_to_delete is not None:
        del pair_array[index_to_delete]
    
    pair_array.append((start_epoch, value))
        
def get_epoch_value(epoch_number, pair_array):
    sorted_pairs = sorted(pair_array)
    cur_value = None
    for pair in sorted_pairs:
        pair_start_epoch = pair[0]
        pair_value = pair[1]
        
        if (pair_start_epoch <= epoch_number):
            cur_value = pair_value
        else:
            break

    return cur_value

class SparseRBMParams(object):

    def __init__(self, num_hidden, num_epochs, learning_rate, gibbs_samples, momentum, weight_penalty):
        self.num_hidden = num_hidden
        self.num_epochs = num_epochs
        
        # We keep an array of pairs (start_epoch, value)
        # At any given time, the value for an epoch is the value from
        # the most recent pair.
        self.learning_rates = [(1, learning_rate)]
        self.gibbs_samples = [(1, gibbs_samples)]
        self.momentums = [(1, momentum)]
        self.weight_penalties = [(1, weight_penalty)]
        
    def set_learning_rate(self, start_epoch, learning_rate):

        assert(start_epoch > 0 and start_epoch <= self.num_epochs)
        set_epoch_value(start_epoch, learning_rate, self.learning_rates)

    def get_learning_rate(self, epoch_number):
        assert(epoch_number > 0 and epoch_number <= self.num_epochs)
        return get_epoch_value(epoch_number, self.learning_rates)

    def set_gibbs_samples(self, start_epoch, gibbs_samples):
        assert(start_epoch > 0 and start_epoch <= self.num_epochs)
        set_epoch_value(start_epoch, gibbs_samples, self.gibbs_samples)

    def get_gibbs_samples(self, epoch_number):
        assert(epoch_number > 0 and epoch_number <= self.num_epochs)
        return get_epoch_value(epoch_number, self.gibbs_samples)
        
    def set_momentum(self, start_epoch, momentum):
        assert(start_epoch > 0 and start_epoch <= self.num_epochs)
        set_epoch_value(start_epoch, momentum, self.momentums)

    def get_momentum(self, epoch_number):
        assert(epoch_number > 0 and epoch_number <= self.num_epochs)
        return get_epoch_value(epoch_number, self.momentums)

    def set_weight_penalty(self, start_epoch, weight_penalty):

        assert(start_epoch > 0 and start_epoch <= self.num_epochs)
        set_epoch_value(start_epoch, weight_penalty, self.weight_penalties)

    def get_weight_penalty(self, epoch_number):
        assert(epoch_number > 0 and epoch_number <= self.num_epochs)
        return get_epoch_value(epoch_number, self.weight_penalties)
    
    def get_epoch_params(self, epoch_number):
        assert(epoch_number > 0 and epoch_number <= self.num_epochs)

        learning_rate = self.get_learning_rate(epoch_number)
        gibbs_samples = self.get_gibbs_samples(epoch_number)
        momentum = self.get_momentum(epoch_number)
        weight_penalty = self.get_weight_penalty(epoch_number)

        return learning_rate, gibbs_samples, momentum, weight_penalty

def main():

    if (len(sys.argv) != 2):
        print "Usage: {0} output_pickle".format(sys.argv[0])
        return 2

    output_pickle = sys.argv[1]
    
    # Generate a basic run profile
    
    sparse_params = SparseRBMParams(20, 40, 0.01, 1, 0.90, 0.0002)
    
    sparse_params.set_gibbs_samples(7, 2)
    sparse_params.set_gibbs_samples(10, 3)
    sparse_params.set_gibbs_samples(21, 4)
    sparse_params.set_gibbs_samples(31, 7)
    sparse_params.set_learning_rate(31, 0.005)

    f = open(output_pickle, 'wb')
    pickle.dump(sparse_params, f)
    f.close()
    
    for epoch in range(sparse_params.num_epochs):
        epoch_num = epoch+1
        alpha, k, momentum, weight_penalty = sparse_params.get_epoch_params(epoch_num)
        print "epoch {}: k={}, alpha={}, momentum={}, weight_penalty={}".format(epoch_num, k, alpha, momentum, weight_penalty)
    
if __name__=="__main__":
    main()
