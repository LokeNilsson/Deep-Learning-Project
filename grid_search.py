from RNN import RNN
from LSTM1 import LSTM1
from LSTM2 import LSTM2
from data_processing import DataManager
import numpy as np
import os


def LSTM2_grid_search(params_dict: dict):
    datamanager = DataManager()
    datamanager.read_files()
    ind_to_char, char_to_ind = datamanager.encode_data()
    
    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(42).state
    
    # Load in set parameters
    epochs = 1
    seq_lengths = params_dict['seq_length']
    m1_list = params_dict['m1']
    m2_list = params_dict['m2']

    # Initialise lists to save results
    grid_search_results = ''
    for seq_length in seq_lengths:
        for m1 in m1_list:
            lstm = None
            for m2 in m2_list:
                # Initialise LSTM
                lstm = LSTM2(m1 = m1, m2=m2, K = datamanager.K, eta = 0.001, rng = rng, tau = seq_length, ind_to_char = ind_to_char, char_to_ind = char_to_ind)
                
                # Divide data in to sequences
                X_train, y_train = datamanager.create_article_sequences(datamanager.training_data, seq_length)
                X_val, y_val = datamanager.create_article_sequences(datamanager.validation_data, seq_length)
                X_test, y_test = datamanager.create_article_sequences(datamanager.test_data, seq_length)

                X_train, y_train, X_val, y_val, X_test, y_test = X_train[0:4], y_train[0:4], X_val[0:10], y_val[0:10], X_test[0:10], y_test[0:10]
                
                # Train network
                training_time, final_val_loss, final_loss = lstm.training(X_train, y_train, X_val, y_val, epochs = epochs)
                
                # Compute test loss
                test_loss = lstm.ComputeLoss(X_test, y_test)
                
                # Save Results
                search_result = f'Sequence Lengths: {seq_length}, m1: {m1}, m2: {m2} \n'
                search_result += f'Training time: {training_time:.2f}    Final Training Loss: {final_loss:.2f}    Final Validation Loss: {final_val_loss:.2f}    Final Test Loss: {test_loss:.2f} \n \n \n \n'
                grid_search_results += search_result
    return grid_search_results


def main():
    # Set Params
    filename = 'test_search'
    # grid_search_parameters = {
    #     'seq_length': [25, 50, 100],
    #     'm1': [50, 100, 150],
    #     'm2': [25, 50, 100],
    #     }
    
    grid_search_parameters = {
        'seq_length': [1, 1, 1],
        'm1': [1, 1, 1],
        'm2': [1, 1, 1],
        }

    # Write Result
    results = LSTM2_grid_search(params_dict = grid_search_parameters)
    with open(f'GridSearch/{filename}.txt', 'w') as f:
        f.write(results)

if __name__== "__main__":
    main()
