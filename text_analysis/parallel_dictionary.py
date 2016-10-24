import sys
import string
import cPickle as pickle
import os
import io
import re
from nltk.tokenize import wordpunct_tokenize
from multiprocessing import Queue
from multiprocessing import Process

# This is a beta version of a dictionary that is able to scan multiple
# files in parallel and combine the dictionaries. This is useful for
# datasets made up of many text files that need to have a shared
# dictionary.
#
# This also implements a pruning algorithm based on dropping the
# least used words. Rather than choose a simple cutoff, the pruning
# function takes in the percent of the dataset that can be cut.
# It then prunes the least used words until that percentage is met.
# This makes the pruning independent of the dataset size, which is
# useful when using a subset for testing.

class Dictionary(object):
    def __init__(self):
        self.word_usage = {}
        self.total_documents = 0
        self.total_words = 0
        
    def merge(self, other_dict):
        for key, value in other_dict.word_usage.iteritems():
            if (key in self.word_usage):
                self.word_usage[key][0] += other_dict.word_usage[key][0]
                self.word_usage[key][1] += other_dict.word_usage[key][1]
            else:
                self.word_usage[key] = other_dict.word_usage[key]
        
        self.total_documents += other_dict.total_documents
        self.total_words += other_dict.total_words

    def process_doc_iterable(self, doc_iterable):
        for document in doc_iterable:
            self.process_document(document)
                
    def process_document(self, document, document_filename=None):
        document_dict = {}

        self.total_documents += 1

        for word in document:
            self.total_words += 1
            
            first_occurrence = False
            if (word not in document_dict):
                document_dict[word] = 1
                first_occurrence = True
            
            # We keep two counts
            # Index 0: count of number of documents this word is in
            # Index 1: count of number of occurrences
            if (word in self.word_usage):
                self.word_usage[word][1] += 1
                if (first_occurrence):
                    self.word_usage[word][0] += 1
            else:
                self.word_usage[word] = [1, 1]

    def prune_least_used_words(self, pct_to_remove):

        assert(pct_to_remove > 0 and pct_to_remove < 100)
        
        # Prune the least words until we reach the pct_to_remove
        total_usages_to_prune = (1.0 * self.total_words * pct_to_remove) / 100

        word_counts = list(self.word_usage.iteritems())

        # Sort by the number of words
        sorted_word_counts = sorted(word_counts, key=lambda x: x[1][1])

        total_usages_deleted = 0
        for word_count in sorted_word_counts:
            word = word_count[0]
            count = word_count[1][1]
            
            if (total_usages_deleted > total_usages_to_prune):
                break
            
            total_usages_deleted += count
            del(self.word_usage[word])

        return total_usages_deleted

def worker_main(task_queue, finish_queue):
    worker_dict = Dictionary()
    words_only_re = re.compile('\w+')
    for document_fname in iter(task_queue.get, 'STOP'):
        with io.open(document_fname, 'r', encoding='utf-8') as f:
            total_doc = []
            for line in f:
                lower_line = line.lower()
                tokens = list(wordpunct_tokenize(lower_line))
                nonnumeric_tokens = []
                for token in tokens:
                    if words_only_re.match(token):
                        nonnumeric_tokens.append(token)
                
                total_doc.extend(nonnumeric_tokens)
            f.close()
            worker_dict.process_document(total_doc, document_fname)
    
    finish_queue.put(worker_dict)

def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) != 3:
        print "Usage: {}: base_dir output_filename".format(argv[0])
        return 2
    
    directory = argv[1]
    print "Input directory: {}".format(directory)
    out_filename = argv[2]
    print "Out Filename: {}".format(out_filename)

    task_queue = Queue()
    finish_queue = Queue() 
    num_threads = 12

    for thread in range(num_threads):
        Process(target=worker_main, args=(task_queue, finish_queue)).start()

    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path):
            task_queue.put(full_path)

    for thread in range(num_threads):
        task_queue.put('STOP')

    central_dict = Dictionary()
    for thread in range(num_threads):
        worker_dict = finish_queue.get()
        central_dict.merge(worker_dict)
        print "Merged worker dictionary into central dictionary"

    print "Total dictionary statistics before prune:"
    print "Number unigrams: {0}".format(len(central_dict.word_usage))
    print "Total words: {0}".format(central_dict.total_words)
    
    uses_pruned = central_dict.prune_least_used_words(1.0)

    print "Uses pruned: {0}".format(uses_pruned)

    print "Total dictionary statistics after prune:"
    print "Number unigrams: {0}".format(len(central_dict.word_usage))
    
    out_file = open(out_filename, "wb")
    pickle.dump(central_dict, out_file)
    out_file.close()

if __name__=="__main__":
    sys.exit(main())
