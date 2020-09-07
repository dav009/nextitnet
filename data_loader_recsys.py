import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn

# This Data_Loader file is copied online
class Data_Loader:

    def fit_transform(self, samples, max_seq_size=80):
        print("samples total")
        print(len(samples))
        new_samples = []
        item_dict = {}
        inverted_item_dict = {}
        m = 1
        for sample in samples:
            for i in sample.split(","):
                if i.strip() not in item_dict:
                    item_dict[i.strip()] = m
                    inverted_item_dict[m] = i.strip()
                    m = m + 1
        for sample in samples:
            s = [item_dict[i.strip()] if i.strip() in item_dict else -1 for i in sample.split(",")]
            if len(s) < max_seq_size:
                pads = [0] * (max_seq_size - len(s))
                s = pads + s
            if len(s)> max_seq_size:
                s = s[-max_seq_size:]
            new_samples.append(s)

        print("new_samples total")
        print(len(new_samples))
        return item_dict, new_samples, inverted_item_dict

            
    def __init__(self, options, max_seq_size=80):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]


        #max_document_length = max([len(x.split(",")) for x in positive_examples])
        #vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        item_dict, new_samples, inverted_item_dict = self.fit_transform(positive_examples, max_seq_size=max_seq_size)

        
        
        self.item = np.array(new_samples)
        self.item_dict = item_dict
        self.vocabulary = inverted_item_dict
        #self.vocabulary = vocab_processor.vocabulary_
        #self.vocab = vocab_processor


    def load_generator_data(self, sample_size):
        text = self.text
        mod_size = len(text) - len(text)%sample_size
        text = text[0:mod_size]
        text = text.reshape(-1, sample_size)
        return text, self.vocab_indexed


    def string_to_indices(self, sentence, vocab):
        indices = [ vocab[s] for s in sentence.split(',') ]
        return indices

    def inidices_to_string(self, sentence, vocab):
        id_ch = { vocab[ch] : ch for ch in vocab } 
        sent = []
        for c in sentence:
            if id_ch[c] == 'eol':
                break
            sent += id_ch[c]

        return "".join(sent)

   
