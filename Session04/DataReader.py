
import numpy as np
import random

class DataReader:
    def __init__(self, data_path, batch_size, vocab_size):
        self._batch_size = batch_size
        self._data = []
        self._labels = []
        self._sentence_lengths = []
        self._num_epoch = 0
        self._batch_id = 0
        with open(data_path) as f:
            d_lines = f.read().splitlines()
        for data_id, line in enumerate(d_lines):
            features = line.split('<fff>')
            label, doc_id, sentence_length = int(features[0]), int(features[1]), int(features[2])
            vector = [int(feature) for feature in features[3].split()]
            self._data.append(vector)
            self._labels.append(label)
            self._sentence_lengths.append(sentence_length)

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._sentence_lengths = np.array(self._sentence_lengths)
        
    def next_batch(self):
        start =  self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id = 0
            indices = list(range(len(self._data)))
            random.seed(2021)
            random.shuffle(indices) 
            self._data, self._labels, self._sentence_lengths \
                = self._data[indices], self._labels[indices], self._sentence_lengths[indices]
        return self._data[start:end], self._labels[start:end], self._sentence_lengths[start:end]
