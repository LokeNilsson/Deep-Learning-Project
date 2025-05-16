import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from collections import Counter
import json
import random

class DataManager():
    def __init__(self):
        # Datasets
        self.all_data = None
        self.training_data = None
        self.validation_data = None
        self.test_data = None

        # Dictionaries
        self.char_to_ind = None
        self.ind_to_char = None

        # Sequence list
        self.seq_list = None

        # variables
        self.K = None

    def read_files(self):
        root_dir = './bbc'
        text_list = []
        for root, _, filenames in os.walk(root_dir):
            if len(filenames) > 0:
                for file in filenames:
                    with open(os.path.join(root, file), 'r') as text_file:
                        text_list.append(text_file.read())

        generator = torch.Generator().manual_seed(42)
        train, val, test = random_split(text_list, [0.7, 0.1, 0.2], generator)
        
        self.all_data = ''.join(text_list)
        self.training_data = ''.join(np.asarray(text_list)[train.indices].tolist())
        self.validation_data = ''.join(np.asarray(text_list)[val.indices].tolist())
        self.test_data = ''.join(np.asarray(text_list)[test.indices].tolist())
        print('Files have been read.')

    def encode_data(self):
        unique_chars = list(set(self.all_data))
        self.K = len(unique_chars)
        # print(''.join(sorted(unique_chars)))
        self.char_to_ind = {}
        self.ind_to_char = {}
        for i, char in enumerate(unique_chars):
            self.char_to_ind[char] = i
            self.ind_to_char[i]    = char 
        print('Data has been encoded')
        return self.ind_to_char, self.char_to_ind

    def create_sequences(self, data:str, seq_length:int):
        # Divide data into sequences
        X_list = []
        y_list = []
        for idx in range(int(len(data) / seq_length)):
            x_seq = data[idx: idx + seq_length]
            y_seq = data[idx + 1: idx + seq_length + 1]
           
            # One-hot encoded input matrix
            X_enc = np.zeros((seq_length, self.K))
            for i, char in enumerate(x_seq):
                ind = self.char_to_ind[char]
                X_enc[i, ind] = 1
            X_list.append(X_enc)

            # ground truth indices
            y_indices = np.zeros(seq_length, dtype = int)
            for i, y_char in enumerate(y_seq):
                y_indices[i] = self.char_to_ind[y_char]
            y_list.append(y_indices)
            
        return X_list, y_list


    def read_HarryPotter(self):
        filepath = f'goblet_book.txt'
        fid = open(filepath, 'r')
        book_data = fid.read()
        fid.close()

        # Extract unique characters from book
        unique_chars = list(set(book_data))
        self.K = len(unique_chars)

        # Create one-hot encoding for every character
        char_to_ind = {}
        ind_to_char = {}
        for idx, char in enumerate(unique_chars): 
            char_to_ind[char] = idx
            ind_to_char[idx] = char
        
        self.char_to_ind = char_to_ind
        self.ind_to_char = ind_to_char
        self.all_data = book_data
        self.K = len(unique_chars)


        self.K = len(unique_chars)
        self.training_data = self.all_data


def main():
    datamanager = DataManager()
    datamanager.read_files()
    datamanager.encode_data()

if __name__== "__main__":
    main()