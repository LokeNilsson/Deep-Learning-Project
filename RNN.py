import torch 
from data_processing import DataManager
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import perf_counter
import os

class RNN:
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

        # bias vectors
        b = np.zeros((1, m), dtype=np.float64)
        c = np.zeros((1, K), dtype=np.float64)

        # weight matrices
        U = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 
        W = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m), dtype=np.float64) 
        V = (1/np.sqrt(m))*rng.standard_normal(size = (m, K), dtype=np.float64) 

        # store parameters in dictionary
        self.rnn = {
            'b' : b,
            'c' : c,
            'U' : U,
            'W' : W,
            'V' : V
        }

        self.m_adam      = {'b': np.zeros_like(b), 'c' : np.zeros_like(c), 'U' : np.zeros_like(U), 'W' : np.zeros_like(W), 'V' : np.zeros_like(V)}
        self.v_adam      = {'b': np.zeros_like(b), 'c' : np.zeros_like(c), 'U' : np.zeros_like(U), 'W' : np.zeros_like(W), 'V' : np.zeros_like(V)}
        self.m_hat_adam  = {'b': np.zeros_like(b), 'c' : np.zeros_like(c), 'U' : np.zeros_like(U), 'W' : np.zeros_like(W), 'V' : np.zeros_like(V)}
        self.v_hat_adam  = {'b': np.zeros_like(b), 'c' : np.zeros_like(c), 'U' : np.zeros_like(U), 'W' : np.zeros_like(W), 'V' : np.zeros_like(V)}
        self.beta1       = 0.9
        self.beta2       = 0.999
        self.eps         = 1e-8


    def ComputeLoss(self, X:list, y:list)->list:
        torch_network = {}
        for kk in self.rnn.keys():
            torch_network[kk] = torch.tensor(self.rnn[kk])

        # Create activation functions       
        apply_tanh = torch.nn.Tanh()
        apply_softmax = torch.nn.Softmax(dim=1) 
        
        loss_list = []
        
        # Iterate over articles
        data = list(zip(X, y))
        hprev = self.last_h
        for Xbatch, ybatch in data:
            Xbatch = torch.from_numpy(Xbatch)
            
            # create an empty tensor to store the hidden vector at each timestep
            Hs = torch.empty(self.tau, self.m, dtype=torch.float64)

            for t in range(self.tau):
                a = torch.matmul(hprev, torch_network['W']) + torch.matmul(Xbatch[t:t+1], torch_network['U']) + torch_network['b']
                assert a.shape == (1,self.m)

                ht = apply_tanh(a)
                Hs[t:t+1, :] = ht

                hprev = ht
            
            Os = torch.matmul(Hs, torch_network['V']) + torch_network['c']       
            P = apply_softmax(Os)    
            
            # compute the loss
            loss = torch.mean(-torch.log(P[np.arange(self.tau), ybatch]))
            loss_list.append(loss.detach().numpy())
            
        return np.mean(loss_list)


    def BackwardsPass(self, X_np:np.ndarray, y_np:np.ndarray, h0_np:np.ndarray) -> dict:
        X = torch.from_numpy(X_np)        
        y = torch.from_numpy(y_np)
        ht = torch.from_numpy(h0_np)
        
        torch_network = {}
        for kk in self.rnn.keys():
            torch_network[kk] = torch.tensor(self.rnn[kk], dtype = torch.float64, requires_grad=True)
     
        apply_tanh = torch.nn.Tanh()
        apply_softmax = torch.nn.Softmax(dim=1)
        dropout = torch.nn.Dropout(p = 0.2)

        # create an empty tensor to store the hidden vector at each timestep
        Hs = torch.empty(X.shape[0], ht.shape[1], dtype=torch.float64)
        
        hprev = ht
        loss_list = []
        for t in range(self.tau):
            a = torch.matmul(hprev, torch_network['W']) + torch.matmul(X[t:t+1,:], torch_network['U']) + torch_network['b']

            ht = apply_tanh(a)
            Hs[t:t+1, :] = ht

            hprev = ht
        
        self.last_h = hprev
        Os = torch.matmul(Hs, torch_network['V']) + torch_network['c']       
        
        # Droput layer
        #Os = dropout(Os)

        P = apply_softmax(Os)    
        
        # compute the loss
        loss = torch.mean(-torch.log(P[np.arange(self.tau), y]))
        loss_list.append(loss.detach().numpy())

        # compute the backward pass relative to the loss and the named parameters 
        loss.backward()

        # extract the computed gradients and make them numpy arrays
        grads = {}
        for kk in self.rnn.keys():
            grads[kk] = torch_network[kk].grad.numpy()


        loss_mean = np.mean(loss_list)
        return grads, loss_mean
    

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

            # reset h between epochs
            self.last_h = torch.zeros(1, self.m, dtype = torch.float64)

            # Iterate over batches
            for Xbatch, ybatch in data:
                    self.tau = Xbatch.shape[0]
                    ht = self.last_h.detach().numpy()
                    # Forward- and backward pass
                    grads, loss = self.BackwardsPass(Xbatch, ybatch, ht)
                    
                    if t == 1:
                        smooth_loss = loss
                        loss_list.append(smooth_loss)
                    else:
                        smooth_loss = 0.999 * smooth_loss + 0.001 *loss 
                        loss_list.append(smooth_loss)

                    # SGD using Adam
                    for kk in grads.keys():
                        self.m_adam[kk]      = self.beta1 * self.m_adam[kk] + (1-self.beta1)*grads[kk]
                        self.v_adam[kk]      = self.beta2 * self.v_adam[kk] + (1-self.beta2)*(grads[kk]**2)
                        self.m_hat_adam[kk]  = self.m_adam[kk]/(1-self.beta1**t)
                        self.v_hat_adam[kk]  = self.v_adam[kk]/(1-self.beta2**t)
                        self.rnn[kk]         = self.rnn[kk] - (self.eta/(np.sqrt(self.v_hat_adam[kk]) + self.eps))*self.m_hat_adam[kk]
                    
                    # Validation
                    if t % 1000 == 0:
                        print(f'iteration: {t}')
                        val_loss.append(self.ComputeLoss(Xval, yval))
                        
                    
                    t += 1 # increment iterations
        # Training time
        end_time = perf_counter()
        self.training_time = end_time - start_time
        print(f'Training took {round(self.training_time, 4)} seconds to execute')
        
        # Plot losses
        self.plot_loss(t, loss_list, val_loss, model_path = model_path)
        return self.rnn

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

        plt.figure('Training Loss', figsize = (10,5))
        plt.plot(t_points, np.asarray(smooth_loss), 'b', label = 'training loss', linewidth = l_width)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xlabel('Update steps', fontsize = f_size)
        plt.ylabel('Smooth loss', fontsize = f_size)
        plt.xlim(0, t)
        plt.ylim(bottom = 0)
        plt.legend(fontsize = f_size)
        if model_path:
            filename = f"{model_path}/training_loss"
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
        self.rnn = model_data['RNN']
        self.char_to_ind = model_data['char_to_ind']
        self.ind_to_char = model_data['ind_to_char']
        self.m = model_data['m']
        self.K = model_data['K']
        self.eta = model_data['eta']
        self.last_h = self.last_h


    def synthesize_text(self, x0:np.ndarray, text_length:int, val_loss = None, T = None, theta = None) -> str:
        # Network Weights and biases

        U, W, V = self.rnn['U'], self.rnn['W'], self.rnn['V']
        b, c = self.rnn['b'], self.rnn['c']

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

                # sorted_p = np.sort(p) # Ascending
                # sorted_p = sorted_p[::-1] # Descending
                
                # pt_sum = np.cumsum(sorted_p)
                # kt = np.argmax(pt_sum >= theta) 
                # p_prim = np.sum(sorted_p[:kt])

                # mask = p >= p[0, kt]
                # p_tilde = np.zeros_like(p)
                # p_tilde[mask] = p[mask] / p_prim

                sorted_inds = np.flip(np.argsort(p, axis=1))[0,:]
                p_sorted = p[:, sorted_inds]
                cum_sum = np.cumsum(p_sorted, axis=1)
                k_t = np.argmax(cum_sum >= theta)
                p_t_k = p_sorted[:, k_t]
                p_prime = np.sum(p_sorted[:, 0:k_t+1])

                mask = np.where(p >= p_t_k, 1, 0)
                p = (p/p_prime)*mask

                # p = p_tilde
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
        if val_loss:
            text_seq += f'\n \n \n \n Validation Loss: {val_loss} \n Training took {self.training_time:.2f} seconds'     
    
        return text_seq    
        
def main():
    datamanager = DataManager()

    # Read BBC articles
    datamanager.read_files()
    ind_to_char, char_to_ind = datamanager.encode_data()

    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(42).state
    
    # Paramaters: ------------------- CHANGE HERE ---------------------------
    seq_length = 25
    m = 100
    epochs = 1
    model_path = f'RNN/m{m}_SL{seq_length}_epochs{epochs}/'
    os.makedirs(os.path.dirname(model_path), exist_ok = True)

    # Initialise RNN
    rnn = RNN(m = m, K = datamanager.K, eta = 0.001, rng = rng, tau = seq_length, ind_to_char = ind_to_char, char_to_ind = char_to_ind)
    
    # Divide data in to sequences
    X_train, y_train = datamanager.create_sequences(datamanager.training_data, seq_length=seq_length)
    X_val, y_val = datamanager.create_sequences(datamanager.validation_data, seq_length=seq_length)
    X_test, y_test = datamanager.create_sequences(datamanager.test_data, seq_length=seq_length)
    print('Sequences created')

    # X_train, y_train, X_val, y_val, X_test, y_test = X_train[0:100], y_train[0:100], X_val[0:10], y_val[0:10], X_test[0:10], y_test[0:10]
    
    # Train network
    rnn.training(X_train, y_train, X_val, y_val, epochs = epochs, model_path = model_path)
    
    # Compute validation loss
    val_loss = rnn.ComputeLoss(X_val, y_val)
    print(f'validation_loss: {round(val_loss, 2)}')
    
    # Generate starting letter
    x0 = np.zeros((1, rnn.K), dtype = np.float64)
    ii = rnn.char_to_ind['T']
    x0[0, ii] = 1
    text_seq = rnn.synthesize_text(x0 = x0, text_length = 1000, val_loss = val_loss)
    with open(f'{model_path}/text.txt', 'w') as f:
        f.write(text_seq)


    rnn.save_model(model_path = model_path)


if __name__== "__main__":
    main()