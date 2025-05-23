import torch 
from data_processing import DataManager
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import perf_counter
import os

class GPT:
    def __init__(self, m, K, eta, rng, tau, ind_to_char, char_to_ind):
        # mapping between characters and integers
        self.ind_to_char = ind_to_char
        self.char_to_ind = char_to_ind
        
        # Networks shapes
        self.m = m
        self.K = K

        # learning rate
        self.eta = eta
        
        # initialize variable to set last h value
        self.last_h = None

        # sequence length
        self.tau = tau
        
        # random generator
        self.rng = rng

        # Self-Attention Layers
        Wq1 = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 
        Wk1 = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 
        Wv1 = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 

        # Transformation Layer
        V1 = (1/np.sqrt(m))*rng.standard_normal(size = (m, m), dtype=np.float64) 
        c1 = np.zeros((1, m), dtype=np.float64)

        # Output Layer
        V2 = (1/np.sqrt(m))*rng.standard_normal(size = (m, K), dtype=np.float64) 
        c2 = np.zeros((1, K), dtype=np.float64)

        # store parameters in dictionary
        self.gpt = {
            # GPT Layer 1
            'Wq1' : Wq1,
            'Wk1' : Wk1,
            'Wv1' : Wv1,
            # Feed Forward layer
            'V1' : V1,
            'c1' : c1,
            # Output Layer
            'V2' : V2,
            'c2' : c2,
        }

        adam_params = {}
        adam_params['beta1'] = 0.9
        adam_params['beta2'] = 0.999
        adam_params['eps'] = 1e-8
        for kk in self.gpt.keys():
            adam_params[f'm_{kk}'] = np.zeros_like(self.gpt[kk])
            adam_params[f'mhat_{kk}'] = np.zeros_like(self.gpt[kk])
            adam_params[f'v_{kk}'] = np.zeros_like(self.gpt[kk])
            adam_params[f'vhat_{kk}'] = np.zeros_like(self.gpt[kk])
        self.adam_params = adam_params


    def ComputeLoss(self, X:list, y:list)->list:
        torch_network = {}
        for kk in self.gpt.keys():
            torch_network[kk] = torch.tensor(self.gpt[kk])

        # Create activation functions       
        apply_softmax = torch.nn.Softmax(dim=1)
        
        loss_list = []

        # Iterate over articles
        data = list(zip(X, y))
        for X_article, y_article in data:
            # sequences from one article
            sequences = list(zip(X_article, y_article))

            for Xbatch_np, ybatch in sequences:
                X = torch.from_numpy(Xbatch_np)
                y = torch.from_numpy(ybatch)
                apply_softmax = torch.nn.Softmax(dim=1)
                apply_relu    = torch.nn.ReLU()

                # calculate query, key and value matrices
                Q1 = torch.matmul(X, torch_network['Wq1'])
                K1 = torch.matmul(X, torch_network['Wk1'])
                V1 = torch.matmul(X, torch_network['Wv1'])

                # get attention weights and multiply with values
                scores = torch.matmul(Q1, K1.T)/torch.sqrt(torch.tensor(self.m))
                upper_tri_inds = torch.triu_indices(scores.shape[0], scores.shape[1], offset=1)
                scores[upper_tri_inds[0], upper_tri_inds[1]] = -torch.inf
                attn_weights1 = apply_softmax(scores)

                Z1 = torch.matmul(attn_weights1, V1)

                # Feedforward       
                FF1 = torch.matmul(Z1, torch_network['V1']) + torch_network['c1'] 

                # RELU
                FF1_relu = apply_relu(FF1)

                # linear layer
                Os = torch.matmul(FF1_relu, torch_network['V2']) + torch_network['c2'] 

                # Softmax to get final predictions
                P = apply_softmax(Os)

                # compute the loss
                loss = torch.mean(-torch.log(P[np.arange(self.tau), y]))
                loss_list.append(loss.detach().numpy())
            
        return np.mean(loss_list)

    def BackwardsPass(self, X_np:np.ndarray, y_np:np.ndarray) -> dict:
        X = torch.from_numpy(X_np)
        y = torch.from_numpy(y_np)

        torch_network = {}
        for kk in self.gpt.keys():
            torch_network[kk] = torch.tensor(self.gpt[kk], dtype = torch.float64, requires_grad=True)

        apply_softmax = torch.nn.Softmax(dim=1)
        apply_relu    = torch.nn.ReLU()

        # calculate query, key and value matrices
        Q1 = torch.matmul(X, torch_network['Wq1'])
        K1 = torch.matmul(X, torch_network['Wk1'])
        V1 = torch.matmul(X, torch_network['Wv1'])

        # get attention weights and multiply with values
        scores = torch.matmul(Q1, K1.T)/torch.sqrt(torch.tensor(self.m))
        upper_tri_inds = torch.triu_indices(scores.shape[0], scores.shape[1], offset=1)
        scores[upper_tri_inds[0], upper_tri_inds[1]] = -torch.inf
        attn_weights1 = apply_softmax(scores)

        Z1 = torch.matmul(attn_weights1, V1)

        # Feedforward       
        FF1 = torch.matmul(torch.nn.LayerNorm(X.shape[-1], bias = False, dtype = torch.float64)(Z1+X), torch_network['V1']) + torch_network['c1'] 

        # RELU
        FF1_relu = apply_relu(FF1)

        # linear layer
        Os = torch.matmul(torch.nn.LayerNorm(Z1.shape[-1], bias = False, dtype = torch.float64)(FF1_relu+Z1), torch_network['V2']) + torch_network['c2'] 

        # Softmax to get final predictions
        P = apply_softmax(Os)

        # compute the loss
        loss = torch.mean(-torch.log(P[np.arange(self.tau), y]))

        # compute the backward pass relative to the loss and the named parameters 
        loss.backward()

        # extract the computed gradients and make them numpy arrays
        grads = {}
        for kk in self.gpt.keys():
            grads[kk] = torch_network[kk].grad.numpy()

        return grads, loss.detach().numpy()
    

    def training(self, X:list, y:list, Xval:np.ndarray, yval:np.ndarray, epochs:int, model_path = None):
        # Initialise lists etc
        loss_list = []
        val_loss = []
        t = 1
        print('Starting training')
        start_time = perf_counter()
        for i in range(epochs): 
            print(f'epoch: {i+1}')
            # Shuffle data
            data = list(zip(X, y))
            self.rng.shuffle(data) 

            # Iterate over articles
            for X_article, y_article in data:
                 # sequences from one article
                sequences = list(zip(X_article, y_article))

                for Xbatch, ybatch in sequences:
                    # Forward- and backward pass
                    grads, loss = self.BackwardsPass(Xbatch, ybatch)

                    # Save loss
                    if t == 1:
                        smooth_loss = loss
                        loss_list.append(smooth_loss)
                    else:
                        smooth_loss = 0.999 * smooth_loss + 0.001 *loss 
                        loss_list.append(smooth_loss)

                    # SGD using Adam
                    for kk in grads.keys():
                        self.adam_params[f'm_{kk}'] = self.adam_params['beta1'] * self.adam_params[f'm_{kk}'] + (1-self.adam_params['beta1'])*grads[kk]
                        self.adam_params[f'v_{kk}'] = self.adam_params['beta2'] * self.adam_params[f'v_{kk}'] + (1 - self.adam_params['beta2']) * grads[kk]**2
                        self.adam_params[f'mhat_{kk}'] = self.adam_params[f'm_{kk}']/(1-self.adam_params['beta1']**t)
                        self.adam_params[f'vhat_{kk}'] = self.adam_params[f'v_{kk}']/(1-self.adam_params['beta2']**t)
            
                        self.gpt[kk] = self.gpt[kk] - (self.eta/(np.sqrt(self.adam_params[f'vhat_{kk}']) + self.adam_params['eps']))*self.adam_params[f'mhat_{kk}']
                    
                    # Validation
                    if t % 10000 == 0:
                        print(f'iteration: {t}')
                        # val_loss.append(self.ComputeLoss(Xval, yval))

                    t += 1
                    
        # Training time
        end_time = perf_counter()
        self.training_time = end_time - start_time
        print(f'Training took {round(self.training_time, 4)} seconds to execute')
        
        # Plot losses
        self.plot_loss(t, loss_list, val_loss, model_path = model_path)
        return self.gpt

    def plot_loss(self, t:int, smooth_loss:list, val_loss:list, model_path = None)->None:

        # Plot Smooth Loss
        f_size = 25
        l_width = 3.0

        t_points = np.linspace(0, t-1, len(smooth_loss))

        plt.figure('Training Loss', figsize = (10,5))
        plt.plot(t_points, np.asarray(smooth_loss), 'b', label = 'training loss', linewidth = l_width)
        if len(val_loss) > 0:
            t_points_val = np.linspace(0, t-1, len(val_loss))
            plt.plot(t_points_val, np.asarray(val_loss), 'r', label = 'validation loss', linewidth=l_width)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xlabel('Update steps', fontsize = f_size)
        plt.ylabel('Smooth loss', fontsize = f_size)
        plt.xlim(0, t)
        plt.ylim(bottom = 0)
        plt.legend(fontsize = f_size)
        if model_path:
            filename = f"{model_path}/loss"
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()


    def save_model(self, model_path):
        model_data = {
            'self': self,
        }
        filename = f"{model_path}/model"
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path):
        with open(f"{model_path}/model", 'rb') as f:
            model_data = pickle.load(f)
        self.gpt = model_data['gpt']
        self.char_to_ind = model_data['char_to_ind']
        self.ind_to_char = model_data['ind_to_char']
        self.m = model_data['m']
        self.K = model_data['K']
        self.eta = model_data['eta']
        self.last_h = self.last_h


    def synthesize_text(self, x0:np.ndarray, text_length:int, test_loss = None, T = None, theta = None) -> str:
        chars = ['T']
        apply_softmax = torch.nn.Softmax(dim=1)
        apply_relu    = torch.nn.ReLU()
        torch_network = {}
        for kk in self.gpt.keys():
            torch_network[kk] = torch.tensor(self.gpt[kk], dtype = torch.float64)

        xt = torch.from_numpy(x0)
        for t in range(text_length):
            # calculate query, key and value matrices
            Q1 = torch.matmul(xt, torch_network['Wq1'])
            K1 = torch.matmul(xt, torch_network['Wk1'])
            V1 = torch.matmul(xt, torch_network['Wv1'])

            # get attention weights and multiply with values
            scores = torch.matmul(Q1, K1.T)/torch.sqrt(torch.tensor(self.m))
            upper_tri_inds = torch.triu_indices(scores.shape[0], scores.shape[1], offset=1)
            scores[upper_tri_inds[0], upper_tri_inds[1]] = -torch.inf
            attn_weights1 = apply_softmax(scores)

            Z1 = torch.matmul(attn_weights1, V1)

            # Feedforward       
            FF1 = torch.matmul(Z1, torch_network['V1']) + torch_network['c1'] 

            # RELU
            FF1_relu = apply_relu(FF1)

            # linear layer
            o_s = torch.matmul(FF1_relu, torch_network['V2']) + torch_network['c2']
            o_s = o_s[-1, :].detach().numpy().reshape(1, -1)
            
            # Check if both T and theta are used
            if T and theta:
                print("You may not use temperature and Nucleus sampling at the same time...")
                return f'do it again...'

            # If T and theta is none, use 
            if T is None and theta is None: 
                p = np.exp(o_s) / np.sum(np.exp(o_s), axis = 1, keepdims = True) # (1, K)    

            elif T and not theta:
                p = np.exp(o_s / T) / np.sum(np.exp(o_s/T), axis = 1, keepdims = True) # (1, K)
            elif not T and theta:
                p = np.exp(o_s) / np.sum(np.exp(o_s), axis = 1, keepdims = True) # (1, K) 

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
            xt_new = torch.zeros(1, self.K, dtype=torch.float64)
            xt_new[0, ii] = 1
            xt = torch.cat([xt, xt_new], dim=0)

        text_seq = "".join(chars)
        if test_loss:
            text_seq += f'\n \n \n \n Test Loss: {test_loss} \n Training took {self.training_time:.2f} seconds'     
    
        return text_seq    
        
def main():
    datamanager = DataManager()

    # Read BBC articles
    datamanager.read_files()
    ind_to_char, char_to_ind = datamanager.encode_data()

    # Read Goblet of Fire --- Needs code adjustments below ---
    # datamanager.read_HarryPotter()
    # ind_to_char, char_to_ind = datamanager.ind_to_char, datamanager.char_to_ind
    # X_val, y_val = None, None
    # test_loss = None

    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(42).state
    
    # Paramaters: ------------------- CHANGE HERE ---------------------------
    seq_length = 25
    m = datamanager.K
    epochs = 200
    model_path = f'GPT/m{m}_SL{seq_length}_epochs{epochs}/'
    os.makedirs(os.path.dirname(model_path), exist_ok = True)

    # Initialise gpt
    gpt = GPT(m = m, K = datamanager.K, eta = 0.0001, rng = rng, tau = seq_length, ind_to_char = ind_to_char, char_to_ind = char_to_ind)
    
    # Divide data in to sequences
    X_train, y_train = datamanager.create_article_sequences(datamanager.training_data, seq_length=seq_length)
    X_val, y_val = datamanager.create_article_sequences(datamanager.validation_data, seq_length=seq_length)
    X_test, y_test = datamanager.create_article_sequences(datamanager.test_data, seq_length=seq_length)
    print('Sequences created')

    X_train, y_train, X_val, y_val, X_test, y_test = X_train[0:1], y_train[0:1], X_val[0:10], y_val[0:10], X_test[0:10], y_test[0:10]
    
    # Train network
    gpt.training(X_train, y_train, X_val, y_val, epochs = epochs, model_path = model_path)
    
    # Compute test loss
    test_loss = gpt.ComputeLoss(X_test, y_test)
    print(f'test loss: {round(test_loss, 2)}')
    
    # Synthesize textproject/gpt.py

    # Generate starting letter
    x0 = np.zeros((1, gpt.K), dtype = np.float64)
    ii = gpt.char_to_ind['T']
    x0[0, ii] = 1
    text_seq = gpt.synthesize_text(x0 = x0, text_length = 1000, test_loss = test_loss)
    with open(f'{model_path}/text.txt', 'w') as f:
        f.write(text_seq)


    gpt.save_model(model_path = model_path)


if __name__== "__main__":
    main()