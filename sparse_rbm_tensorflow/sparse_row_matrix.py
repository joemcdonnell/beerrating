import numpy as np
import sys
import math
import string

# This is a modified version of the CSR format.
# For a description of CSR, see https://en.wikipedia.org/wiki/Sparse_matrix
#
# CSR has the following arrays:
# ratings array
# column index array
# row extents array
#
# The typical format for the row extents array is:
# [0, start row 2, start row 3, ..., start of row N, max idx]
#
# Each row's extents are determined by adjacent elements in the array.
# Two adjacent elements with the same value indicate an empty row.
# Otherwise, it indicates the start index and end index (not inclusive).
#
# i.e.
# Row 1 is [0, start row 2]
# Row 2 is [start row 2, start row 3]
# Row 3 is [start row 3, start row 4]
# And so forth.
#
# I'm interested in only rows that are not all zero. I have no operations
# to perform on all-zero rows. So, the modification is that the row
# extents are stored as a [start index, size] pair. Elements that would
# have a size of zero are omitted. 
#
#
# Usage:
# This takes in a text file in the coordinate list format. Specifically,
# it follows the following format:
# ${user_index}\t${item_index}\t${rating}
#
# The file should be sorted by user_index and then by item_index for each
# user_index.
#
# This file generates some aggregated statistics that are useful. It
# generates an overall average of all the ratings. It also generates
# a per-item average rating. It maintains a dictionary that tracks
# which user indexes are present.
#
# It maintains a rowidx to useridx map along with the reverse.

class CompressedSparseRow(object):

    def __init__(self, sparse_matrix_file=None):
        self.max_user_idx = 0
        self.max_item_idx = 0
        self.useridx_2_rowidx = {}
        self.rowidx_2_useridx = {}
        self.num_rows = 0
        self.indexes_array = None
        self.ratings_array = None
        self.row_extents_array = None
        self.item_averages = {}
        self.user_dict = {}
        self.overall_average = 0.0

        if (sparse_matrix_file is not None):
            self.read_matrix_file(sparse_matrix_file)

    def __str__(self):
        s = "num_rows: {}, num_items: {}, max_user_idx: {}, max_item_idx: {}\n"
        return s.format(self.num_rows, len(self.ratings_array), self.max_user_idx, self.max_item_idx)
            
    def read_matrix_file(self, sparse_matrix_file):
        
        self.item_averages = {}
        self.user_averages = {}
        
        # First pass: generate per-item averages and overall average
        # Also, keep track of the users that have reviews
        total_sum = 0.0
        total_count = 0
        with open(sparse_matrix_file, 'r') as f:
            for line in f:
                fields = string.split(line, '\t')
                user_index = int(fields[0])
                item_index = int(fields[1])
                rating = float(fields[2])

                if (user_index > self.max_user_idx):
                    self.max_user_idx = user_index
                    
                if (item_index > self.max_item_idx):
                    self.max_item_idx = item_index
                
                total_sum += rating
                total_count += 1
                if (item_index in self.item_averages):
                    item_entry = self.item_averages[item_index]
                    cur_sum = item_entry[0]
                    count = item_entry[1]
                    self.item_averages[item_index] = [cur_sum + rating, count+1]
                else:
                    self.item_averages[item_index] = [rating, 1]
        
                if (user_index not in self.user_dict):
                    self.user_dict[user_index] = 1

            f.close()

        self.overall_average = (1.0 * total_sum) / total_count

                
        # Allocate arrays
        # Ratings array has size of total_count
        # Item indexes array has size of total_count
        # Row extents array has size 2 * len(self.user_dict)

        num_rows = len(self.user_dict)
        
        self.indexes_array = np.zeros(total_count, dtype=np.int32)
        self.ratings_array = np.zeros(total_count, dtype=np.float32)
        self.row_extents_array = np.zeros((num_rows,2), dtype=np.int32)
        self.useridx_2_rowidx = np.zeros(self.max_user_idx + 1, dtype=np.int32)
        self.rowidx_2_useridx = np.zeros(num_rows, dtype=np.int32)
        self.num_rows = num_rows

        # Second pass: fill in the arrays
        cur_user = None
        row_idx = 0
        start = 0
        size = 0
        with open(sparse_matrix_file, 'r') as f:
            for idx, line in enumerate(f):
                fields = string.split(line, '\t')
                user_index = int(fields[0])
                item_index = int(fields[1])
                rating = float(fields[2])   

                if (cur_user is None):
                    cur_user = user_index

                self.ratings_array[idx] = rating
                self.indexes_array[idx] = item_index

                if user_index != cur_user:
                    # Maintain maps row_idx <-> user_idx
                    self.useridx_2_rowidx[cur_user] = row_idx
                    self.rowidx_2_useridx[row_idx] = cur_user
                    self.row_extents_array[row_idx] = np.array([start, size])

                    # Make sure everything is sorted
                    l = self.indexes_array[start:start+size]
                    if (not all(l[i] < l[i+1] for i in xrange(len(l)-1))):
                        print "not sorted"
                        print self.indexes_array[start:start+size]
                    cur_user = user_index
                    row_idx += 1
                    start += size
                    size = 1
                else:
                    size += 1
            
            f.close()

def main():

    if len(sys.argv) != 2:
        print "Usage: {0} sparse_matrix_file".format(sys.argv[0])
        return 2

    sparse_matrix_file = sys.argv[1]

    csr = CompressedSparseRow(sparse_matrix_file)
    

if __name__=="__main__":
    main()
