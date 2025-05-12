import os
import pandas as pd
import torch
from torch.utils.data.dataset import random_split
from collections import Counter
import json
import random

class DataManager():
    def __init__(self):
        
        # Datasets
        self.df_data = None
        self.training_data = None
        self.validation_data = None
        self.test_data = None

        # Dictionaries
        self.char_to_ind = None
        self.ind_to_char = None
    
    def read_files(self):
        df = pd.read_csv("bbc-text.csv")
        df.fillna('', inplace=True)

        generator = torch.Generator().manual_seed(42)
        train, val, test = random_split(df['text'], [0.7,0.1, 0.2], generator)
        
        self.df_data = df['text'].to_string(index = False, header = False)
        self.training_data = df['text'][train.indices]
        self.validation_data = df['text'][val.indices]
        self.test_data = df['text'][test.indices]

        print('Files have been read.')


    def encode_data(self):
        unique_chars = list(set(self.df_data))
        print(any(c.isupper() for c in self.df_data))
        self.K = len(unique_chars)
        print(''.join(unique_chars))
        char_to_ind = {}
        ind_to_char = {}
        for i, char in enumerate(unique_chars):
            char_to_ind[char] = i
            ind_to_char[i]    = char 
        print(self.K)
        print('Data has been encoded')
        

    def create_sequences(self, seq_length:int):
        seq_length = 25

        # Divide data into sequences
        seq_list = []
        for idx in range(int(len(self.df_data) / seq_length)):
            x_seq = self.df_data[idx: idx + seq_length]
            y_seq = self.df_data[idx + 1: idx + seq_length + 1]
            seq_list.append((x_seq, y_seq))
        


def main():
    datamanager = DataManager()
    datamanager.read_files()
    datamanager.encode_data()

if __name__== "__main__":
    main()