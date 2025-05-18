import torch 
from data_processing import DataManager
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import perf_counter
import os

class LSTM1:
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

        # Cell state
        self.ct_prev = torch.zeros(1, m, dtype=torch.float64)

        # Input gate
        Wix = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 
        Wih = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m), dtype=np.float64) 
        bi = np.zeros((1, m), dtype=np.float64)

        Wcx = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64)
        Wch = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m), dtype=np.float64) 
        bc = np.zeros((1, m), dtype=np.float64)

        # Forget gate
        Wfx =  (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64)
        Wfh = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m), dtype=np.float64) 
        bf = np.zeros((1, m), dtype=np.float64)

        # Output gate
        Wox = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64)
        Woh = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m), dtype=np.float64)
        bo = np.zeros((1, m), dtype=np.float64)
        
        # Dense output layer
        c = np.zeros((1, K), dtype=np.float64)
        V = (1/np.sqrt(m))*rng.standard_normal(size = (m, K), dtype=np.float64)
        
        # store parameters in dictionary
        self.lstm = {
            'Wix': Wix,
            'Wih': Wih,
            'Wcx': Wcx,
            'Wch': Wch, 
            'Wfx': Wfx,
            'Wfh': Wfh,
            'Wox': Wox,
            'Woh': Woh,
            'bi': bi,
            'bc': bc,
            'bf': bf,
            'bo': bo,
            'V': V,
            'c': c
        }
        
        adam_params = {}
        adam_params['beta1'] = 0.9
        adam_params['beta2'] = 0.999
        adam_params['eps'] = 1e-8
        for kk in self.lstm.keys():
            adam_params[f'm_{kk}'] = np.zeros_like(self.lstm[kk])
            adam_params[f'mhat_{kk}'] = np.zeros_like(self.lstm[kk])
            adam_params[f'v_{kk}'] = np.zeros_like(self.lstm[kk])
            adam_params[f'vhat_{kk}'] = np.zeros_like(self.lstm[kk])
        self.adam_params = adam_params

    def ComputeLoss(self, X:list, y:list)->list:        
        tau = self.tau

        torch_network = {}
        for kk in self.lstm.keys():
            torch_network[kk] = torch.tensor(self.lstm[kk])

        # Create activation functions       
        apply_tanh = torch.nn.Tanh()
        apply_softmax = torch.nn.Softmax(dim=1) 
        apply_sigmoid = torch.nn.Sigmoid()
        
        loss_list = []
        hprev = self.last_h.detach()
        ct = self.ct_prev.detach()
        # Iterate over articles
        data = list(zip(X, y))
        hprev = self.last_h
        for Xbatch, ybatch in data:
            Xbatch = torch.from_numpy(Xbatch)
            
            # create an empty tensor to store the hidden vector at each timestep
            Hs = torch.empty(self.tau, self.m, dtype=torch.float64)

            for t in range(self.tau):
                # input gate
                it = apply_sigmoid(torch.matmul(Xbatch[t:t+1,:], torch_network['Wix']) + torch.matmul(hprev, torch_network['Wih']) + torch_network['bi'])

                # candidate input
                c_tilde = apply_tanh(torch.matmul(Xbatch[t:t+1,:], torch_network['Wcx']) + torch.matmul(hprev, torch_network['Wch']) + torch_network['bc'])

                # forget gate
                ft = apply_sigmoid(torch.matmul(Xbatch[t:t+1,:], torch_network['Wfx']) + torch.matmul(hprev, torch_network['Wfh']) + torch_network['bf'])

                # update cell state
                ct = ft*ct + it*c_tilde

                # output gate
                ot = apply_sigmoid(torch.matmul(Xbatch[t:t+1,:], torch_network['Wox']) + torch.matmul(hprev, torch_network['Woh']) + torch_network['bo'])

                # update hidden state (long-term memory)
                ht = ot*apply_tanh(ct)
                Hs[t:t+1, :] = ht

                hprev = ht
            
            P = apply_softmax(torch.matmul(Hs, torch_network['V']) + torch_network['c']  )    
            
            # compute the loss
            loss = torch.mean(-torch.log(P[np.arange(tau), ybatch]))
            loss_list.append(loss.detach().numpy())
        
        return np.mean(loss_list)


    def BackwardsPass(self, X_np:np.ndarray, y_np:np.ndarray, h0_np:np.ndarray) -> dict:
        X = torch.from_numpy(X_np)        
        y = torch.from_numpy(y_np)
        ht = torch.from_numpy(h0_np)
        
        torch_network = {}
        for kk in self.lstm.keys():
            torch_network[kk] = torch.tensor(self.lstm[kk], dtype = torch.float64, requires_grad=True)
     
        apply_tanh = torch.nn.Tanh()
        apply_softmax = torch.nn.Softmax(dim = 1) 
        apply_sigmoid = torch.nn.Sigmoid()
        
        # create an empty tensor to store the hidden vector at each timestep
        Hs = torch.empty(X.shape[0], ht.shape[1], dtype = torch.float64)
        
        hprev = ht
        loss_list = []
        ct = self.ct_prev.detach()
        for t in range(self.tau):
            # input gate
            it = apply_sigmoid(torch.matmul(X[t:t+1,:], torch_network['Wix']) + torch.matmul(hprev, torch_network['Wih']) + torch_network['bi'])

            # candidate input
            c_tilde = apply_tanh(torch.matmul(X[t:t+1,:], torch_network['Wcx']) + torch.matmul(hprev, torch_network['Wch']) + torch_network['bc'])

            # forget gate
            ft = apply_sigmoid(torch.matmul(X[t:t+1,:], torch_network['Wfx']) + torch.matmul(hprev, torch_network['Wfh']) + torch_network['bf'])

            # update cell state
            ct = ft*ct + it*c_tilde

            # output gate
            ot = apply_sigmoid(torch.matmul(X[t:t+1,:], torch_network['Wox']) + torch.matmul(hprev, torch_network['Woh']) + torch_network['bo'])

            # update hidden state (long-term memory)
            ht = ot*apply_tanh(ct)
            Hs[t:t+1, :] = ht

            hprev = ht
        
        self.last_h = hprev 
        self.ct_prev = ct       
        
        P = apply_softmax(torch.matmul(Hs, torch_network['V']) + torch_network['c'])    
        
        # compute the loss
        loss = torch.mean(-torch.log(P[np.arange(self.tau), y]))
        loss_list.append(loss.detach().numpy())

        # compute the backward pass relative to the loss and the named parameters 
        loss.backward()

        # extract the computed gradients and make them numpy arrays
        grads = {}
        for kk in self.lstm.keys():
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
                ht = self.last_h.detach().numpy()

                # Forward- and backward pass
                grads, loss = self.BackwardsPass(Xbatch, ybatch, ht)

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
        
                    self.lstm[kk] = self.lstm[kk] - (self.eta/(np.sqrt(self.adam_params[f'vhat_{kk}']) + self.adam_params['eps']))*self.adam_params[f'mhat_{kk}']
                
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
        return self.lstm

    def plot_loss(self, t:int, smooth_loss:list, val_loss:list, model_path = None)->None:

        # Plot Smooth Loss
        f_size = 25
        l_width = 3.0

        t_points = np.linspace(0, t-1, len(smooth_loss))

        plt.figure('Loss', figsize = (10,5))
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
            'self' : self,
        }
        filename = f"{model_path}/model"
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)


    def synthesize_text(self, x0:np.ndarray, text_length:int, val_loss = None, T = None, theta = None) -> str:
        chars = ['T']
        xt = torch.from_numpy(x0)

        # Load net
        torch_network = {}
        for kk in self.lstm.keys():
            torch_network[kk] = torch.tensor(self.lstm[kk], dtype = torch.float64, requires_grad=True)
     
        apply_tanh = torch.nn.Tanh()
        apply_sigmoid = torch.nn.Sigmoid()

        hprev = self.last_h.detach()
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
            hprev = ht


            o_s = torch.matmul(ht, torch_network['V']) + torch_network['c']
            o_s = o_s.detach().numpy()
            
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
            xt = torch.zeros(1, self.K, dtype=torch.float64)
            xt[0, ii] = 1
            
        text_seq = "".join(chars)
        if val_loss:
            text_seq += f'\n \n \n \n Validation Loss: {val_loss} \n Training took {self.training_time:.2f} seconds'   
    
        return text_seq
        
        
def main():
    datamanager = DataManager()
    datamanager.read_files()
    ind_to_char, char_to_ind = datamanager.encode_data()
    
    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(42).state
    
    # Parameters: ------------------- CHANGE HERE ---------------------------
    m = 100
    seq_length = 100
    eta = 0.001

    epochs = 1
    model_path = f'LSTM1/m{m}_SL{seq_length}_epochs{epochs}_eta{eta}/'
    os.makedirs(os.path.dirname(model_path), exist_ok = True)

    # Initialise LSTM
    lstm = LSTM1(m = m, K = datamanager.K, eta = eta, rng = rng, tau = seq_length, ind_to_char = ind_to_char, char_to_ind = char_to_ind)
    
    # Divide data in to sequences
    X_train, y_train = datamanager.create_sequences(datamanager.training_data, seq_length)
    X_val, y_val = datamanager.create_sequences(datamanager.validation_data, seq_length)
    X_test, y_test = datamanager.create_sequences(datamanager.test_data, seq_length)
    print('Sequences created')

    # X_train, y_train, X_val, y_val, X_test, y_test = X_train[0:100], y_train[0:100], X_val[0:10], y_val[0:10], X_test[0:10], y_test[0:10]

    # Train network
    lstm.training(X_train, y_train, X_val, y_val, epochs = epochs, model_path = model_path)
    
    # Compute test loss
    val_loss = lstm.ComputeLoss(X_val, y_val)
    print(f'validation loss: {round(val_loss, 2)}')
    
    # Synthesize text
    # Generate starting letter
    x0 = np.zeros((1, lstm.K), dtype = np.float64)
    ii = lstm.char_to_ind['T']
    x0[0, ii] = 1
    text_seq = lstm.synthesize_text(x0 = x0, text_length = 1000, val_loss = val_loss)
    with open(f'{model_path}/text.txt', 'w') as f:
        f.write(text_seq)

    lstm.save_model(model_path = model_path)


if __name__== "__main__":
    main()