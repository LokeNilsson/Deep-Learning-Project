from RNN import RNN
from LSTM1 import LSTM1
from LSTM2 import LSTM2
import pickle
from spellchecker import SpellChecker
import numpy as np
import random
import torch

class ModelEvaluator():
    def __init__(self, model_type: str):
        self.spellchecker = SpellChecker()

        self.model_type = model_type
        
        # Randomiser
        rng = np.random.default_rng()
        BitGen = type(rng.bit_generator)
        rng.bit_generator.state = BitGen(42).state
        self.rng = rng



    def load_net(self, model_path, model_archetype):
        with open(f"{model_path}/model", 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data[model_archetype]
        self.char_to_ind = model_data['char_to_ind']
        self.ind_to_char = model_data['ind_to_char']
        self.m = model_data['m']
        self.K = model_data['K']
        self.eta = model_data['eta']
        self.last_h = model_data['last_h']
        
    
    def RNN_synthesize_text(self, x0:np.ndarray, text_length:int, model_path = None, test_loss = None, T = None, theta = None) -> str:
        # Network Weights and biases
        rnn = self.model

        U, W, V = rnn['U'], rnn['W'], rnn['V']
        b, c = rnn['b'], rnn['c']

        h = self.last_h.detach().numpy()
        chars = []

        xt = x0.reshape(1, self.K) # (1, K)
        for t in range(text_length):
            
            a = xt@U + h@W + b              # (1, m)
            h = np.tanh(a)                  # (1, m)
            o = h @ V + c                   # (1, K)
            
            # Check if both T and theta are used
            if T and theta:
                print("You may not use temperature and Nucleus sampling at the same time...")
                return f'do it again...'

            # If T and theta is none, use 
            if T is None and theta is None: 
                p = np.exp(o) / np.sum(np.exp(o), axis = 1, keepdims = True) # (1, K)    

            elif T and not theta:
                p = np.exp(o / T) / np.sum(np.exp(o/T), axis = 1, keepdims = True) # (1, K)
            elif not T and theta:
                p = np.exp(o) / np.sum(np.exp(o), axis = 1, keepdims = True) # (1, K) 

                sorted_p = np.sort(p) # Ascending
                sorted_p = sorted_p[::-1] # Descending
                
                pt_sum = np.cumsum(sorted_p)
                kt = np.argmax(pt_sum >= theta) 
                p_prim = np.sum(sorted_p[:kt])

                mask = p >= p[0, kt]
                p_tilde = np.zeros_like(p)
                p_tilde[mask] = p[mask] / p_prim

                p = p_tilde
            else: print("Error: Can not use temperature and nucleus sampling at the same time...")

            p = p.flatten()
            cp = np.cumsum(p, axis = 0)
            a = self.rng.uniform(size = 1)
            ii = np.argmax(cp - a > 0)
            chars.append(self.ind_to_char[ii])
            
            # Set sampled charcter to next input
            xt = np.zeros((1, self.K))
            xt[0, ii] = 1

        text_seq = "".join(chars)
        return text_seq


    def LSTM_synthesize_text(self, x0:np.ndarray, text_length:int, model_path = None, test_loss = None, T = None, theta = None) -> str:
        chars = []
        lstm = self.model

        # Load net
        torch_network = {}
        for kk in lstm.keys():
            torch_network[kk] = torch.tensor(lstm[kk], dtype = torch.float64, requires_grad=True)
     
        apply_tanh = torch.nn.Tanh()
        apply_sigmoid = torch.nn.Sigmoid()

        hprev = self.last_h
        xt = torch.from_numpy(x0)
        ct = self.ct_prev.detach()
        for t in range(text_length):
            # input gate
            it = apply_sigmoid(torch.matmul(xt, torch_network['Wix']) + torch.matmul(hprev, torch_network['Wih']) + torch_network['bi'])

            # candidate input
            c_tilde = apply_tanh(torch.matmul(xt, torch_network['Wcx']) + torch.matmul(hprev, torch_network['Wch']) + torch_network['bc'])

            # forget gate
            ft = apply_sigmoid(torch.matmul(xt, torch_network['Wfx']) + torch.matmul(hprev, torch_network['Wfh']) + torch_network['bf'])

            # update cell state
            ct = ft*ct + it*c_tilde

            # output gate
            ot = apply_sigmoid(torch.matmul(xt, torch_network['Wox']) + torch.matmul(hprev, torch_network['Woh']) + torch_network['bo'])

            # update hidden state (long-term memory)
            ht = ot*apply_tanh(ct)

            ot = ot.detach().numpy()
            hprev = ht
            
            # Check if both T and theta are used
            if T and theta:
                print("You may not use temperature and Nucleus sampling at the same time...")
                return f'do it again...'

            # If T and theta is none, use 
            if T is None and theta is None: 
                p = np.exp(ot) / np.sum(np.exp(ot), axis = 1, keepdims = True) # (1, K)    

            elif T and not theta:
                p = np.exp(ot / T) / np.sum(np.exp(ot/T), axis = 1, keepdims = True) # (1, K)
            elif not T and theta:
                p = np.exp(ot) / np.sum(np.exp(ot), axis = 1, keepdims = True) # (1, K) 

                sorted_p = np.sort(p) # Ascending
                sorted_p = sorted_p[::-1] # Descending
                
                pt_sum = np.cumsum(sorted_p)
                kt = np.argmax(pt_sum >= theta) 
                p_prim = np.sum(sorted_p[:kt])

                mask = p >= p[0, kt]
                p_tilde = np.zeros_like(p)
                p_tilde[mask] = p[mask] / p_prim

                p = p_tilde
            else: print("Error: Can not use temperature and nucleus sampling at the same time...")

            p = p.flatten()
            cp = np.cumsum(p, axis = 0)
            a = self.rng.uniform(size = 1)
            ii = np.argmax(cp - a > 0)
            chars.append(self.ind_to_char[ii])
            
            # Set sampled charcter to next input
            xt = torch.zeros(1, self.K, dtype=torch.float64)
            xt[0, ii] = 1
            
        text_seq = "".join(chars)
        return text_seq
    


    def spellcheck(self, text):
        # Filter for words
        filtered_all_the_words = True

        # Convert to list
        list_of_words = []

        # Total Amount of words
        tot_words = len(list_of_words)

        # Spell check
        misspelled = self.spellchecker.unknown(list_of_words) # int?
        spelling_accuracy = misspelled / tot_words
        
        return tot_words, spelling_accuracy

def main():
    # Paramteres to fill in:
    model_path = 'RNN/m50_SL25_epochs1'
    model_type = 'RNN'
    text_length = 1000

    # Initialise Evaluator
    evaluator = ModelEvaluator(model_type = model_type)
    evaluator.load_net(model_path = model_path, model_archetype = model_type)   
    
    # Random starting character
    all_char_indices = list(evaluator.char_to_ind.values())
    start_char_idx = random.choice(all_char_indices)
    x0 = np.zeros((1, evaluator.K))
    print(type(start_char_idx))
    x0[0, start_char_idx] = 1
    
    # Generate Text
    generated_text = evaluator.RNN_synthesize_text(x0 = x0, text_length = text_length, model_path = None, test_loss = None, T = None, theta = None)

    # Check accuracy
    total_words, spelling_accuracy = evaluator.spellcheck(generated_text)
    print(f'{model_type} model generated {total_words} with an accuracy of {spelling_accuracy:.2f} %')

    



if __name__== "__main__":
    main()