import torch 
from data_processing import DataManager
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import perf_counter
import os

class LSTM2:
    def __init__(self, m1, m2, K, eta, rng, tau, ind_to_char, char_to_ind):
        # mapping between characters and integers
        self.ind_to_char = ind_to_char
        self.char_to_ind = char_to_ind
        
        # Networks shapes
        self.m1 = m1
        self.m2 = m2
        
        self.K = K

        # learning rate
        self.eta = eta
        
        # initialize variable to set last h value
        self.last_h = None

        # sequence length
        self.tau = tau
        
        # random generator
        self.rng = rng 
        
        # ------ Layer 1 -------
        # Cell state
        self.ct_prev1 = torch.zeros(1, m1, dtype=torch.float64)
        self.ct_prev2 = torch.zeros(1, m2, dtype=torch.float64)
        
        # Input gate
        Wix1 = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m1), dtype=np.float64) 
        Wih1 = (1/np.sqrt(2*m1))*rng.standard_normal(size = (m1, m1), dtype=np.float64) 
        bi1 = np.zeros((1, m1), dtype=np.float64)

        Wcx1 = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m1), dtype=np.float64)
        Wch1 = (1/np.sqrt(2*m1))*rng.standard_normal(size = (m1, m1), dtype=np.float64) 
        bc1 = np.zeros((1, m1), dtype=np.float64)

        # Forget gate
        Wfx1 =  (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m1), dtype=np.float64)
        Wfh1 = (1/np.sqrt(2*m1))*rng.standard_normal(size = (m1, m1), dtype=np.float64) 
        bf1 = np.zeros((1, m1), dtype=np.float64)

        # Output gate
        Wox1 = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m1), dtype=np.float64)
        Woh1 = (1/np.sqrt(2*m1))*rng.standard_normal(size = (m1, m1), dtype=np.float64)
        bo1 = np.zeros((1, m1), dtype=np.float64)

        # ----- Layer 2 ------
        # Input gate
        Wix2 = (1/np.sqrt(2*m1))*rng.standard_normal(size = (m1, m2), dtype=np.float64) 
        Wih2 = (1/np.sqrt(2*m2))*rng.standard_normal(size = (m2, m2), dtype=np.float64) 
        bi2 = np.zeros((1, m2), dtype=np.float64)

        Wcx2 = (1/np.sqrt(2*m1))*rng.standard_normal(size = (m1, m2), dtype=np.float64)
        Wch2 = (1/np.sqrt(2*m2))*rng.standard_normal(size = (m2, m2), dtype=np.float64) 
        bc2 = np.zeros((1, m2), dtype=np.float64)

        # Forget gate
        Wfx2 =  (1/np.sqrt(2*m1))*rng.standard_normal(size = (m1, m2), dtype=np.float64)
        Wfh2 = (1/np.sqrt(2*m2))*rng.standard_normal(size = (m2, m2), dtype=np.float64) 
        bf2 = np.zeros((1, m2), dtype=np.float64)

        # Output gate
        Wox2 = (1/np.sqrt(2*m1))*rng.standard_normal(size = (m1, m2), dtype=np.float64)
        Woh2 = (1/np.sqrt(2*m2))*rng.standard_normal(size = (m2, m2), dtype=np.float64)
        bo2 = np.zeros((1, m2), dtype=np.float64)
    
        # ----- Dense output layer -----
        c = np.zeros((1, K), dtype=np.float64)
        V = (1/np.sqrt(m2))*rng.standard_normal(size = (m2, K), dtype=np.float64)
        
        # store parameters in dictionary
        self.lstm = {
            'Wix1': Wix1, 'Wix2': Wix2,
            'Wih1': Wih1, 'Wih2': Wih2,
            'Wcx1': Wcx1, 'Wcx2': Wcx2,
            'Wch1': Wch1, 'Wch2': Wch2,
            'Wfx1': Wfx1, 'Wfx2': Wfx2,
            'Wfh1': Wfh1, 'Wfh2': Wfh2,
            'Wox1': Wox1, 'Wox2': Wox2,
            'Woh1': Woh1, 'Woh2': Woh2,
            'bi1' : bi1,   'bi2': bi2,
            'bc1' : bc1,   'bc2': bc2,
            'bf1' : bf1,   'bf2': bf2,
            'bo1' : bo1,   'bo2': bo2,
            'V'   : V,       'c': c
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

    def ComputeLoss(self, X:list, y:list) -> list:
        tau = self.tau

        torch_network = {}
        for kk in self.lstm.keys():
            torch_network[kk] = torch.tensor(self.lstm[kk], requires_grad=True)

        # Create activation functions       
        apply_tanh = torch.nn.Tanh()
        apply_softmax = torch.nn.Softmax(dim=1) 
        apply_sigmoid = torch.nn.Sigmoid()
        
        loss_list = []
        
        
        ct1 = self.ct_prev1.detach()
        ct2 = self.ct_prev2.detach()
        data = list(zip(X, y))
        for X_article, y_article in data:
            # sequences from one article
            sequences = list(zip(X_article, y_article))

            hprev1, hprev2 = self.last_h1.detach(), self.last_h2.detach()
            for Xbatch_np, ybatch in sequences:
                Xbatch = torch.from_numpy(Xbatch_np)

                # create an empty tensor to store the hidden vector at each timestep
                Hs = torch.empty(self.tau, self.m2, dtype=torch.float64) 

                for t in range(self.tau):
                    # input gate
                    it1 = apply_sigmoid(torch.matmul(Xbatch[t:t+1, :], torch_network['Wix1']) + torch.matmul(hprev1, torch_network['Wih1']) + torch_network['bi1'])

                    # candidate input
                    c_tilde1 = apply_tanh(torch.matmul(Xbatch[t:t+1, :], torch_network['Wcx1']) + torch.matmul(hprev1, torch_network['Wch1']) + torch_network['bc1'])

                    # forget gate
                    ft2 = apply_sigmoid(torch.matmul(Xbatch[t:t+1], torch_network['Wfx1']) + torch.matmul(hprev1, torch_network['Wfh1']) + torch_network['bf1'])

                    # update cell state
                    ct1 = ft2*ct1 + it1*c_tilde1

                    # output gate
                    ot1 = apply_sigmoid(torch.matmul(Xbatch[t:t+1], torch_network['Wox1']) + torch.matmul(hprev1, torch_network['Woh1']) + torch_network['bo1'])

                    # update hidden state (long-term memory)
                    ht1 = ot1*apply_tanh(ct1)
                    hprev1 = ht1

                    x2 = ht1
                    assert x2.shape == (1, self.m1)
                    # input gate
                    it2 = apply_sigmoid(torch.matmul(x2, torch_network['Wix2']) + torch.matmul(hprev2, torch_network['Wih2']) + torch_network['bi2'])

                    # candidate input
                    c_tilde2 = apply_tanh(torch.matmul(x2, torch_network['Wcx2']) + torch.matmul(hprev2, torch_network['Wch2']) + torch_network['bc2'])

                    # forget gate
                    ft2 = apply_sigmoid(torch.matmul(x2, torch_network['Wfx2']) + torch.matmul(hprev2, torch_network['Wfh2']) + torch_network['bf2'])

                    # update cell state
                    ct2 = ft2*ct2 + it2*c_tilde2
                    assert ct2.shape == (1, self.m2)

                    # output gate
                    ot2 = apply_sigmoid(torch.matmul(x2, torch_network['Wox2']) + torch.matmul(hprev2, torch_network['Woh2']) + torch_network['bo2'])
                    assert ot2.shape == (1, self.m2)
                    ht2 = ot2*apply_tanh(ct2)
                    hprev2 = ht2

                    Hs[t:t+1, :] = ht2
                
                P = apply_softmax(torch.matmul(Hs, torch_network['V']) + torch_network['c']  )    
                
                # compute the loss
                loss = torch.mean(-torch.log(P[np.arange(tau), ybatch]))
                loss_list.append(loss.detach().numpy())
        
        return np.mean(loss_list)


    def BackwardsPass(self, X_np:np.ndarray, y_np:np.ndarray, h0_1:np.ndarray, h0_2:np.ndarray) -> dict:
        X = torch.from_numpy(X_np)        
        y = torch.from_numpy(y_np)
        hprev1 = torch.from_numpy(h0_1)
        hprev2 = torch.from_numpy(h0_2)
        
        torch_network = {}
        for kk in self.lstm.keys():
            torch_network[kk] = torch.tensor(self.lstm[kk], dtype = torch.float64, requires_grad=True)
     
        apply_tanh = torch.nn.Tanh()
        apply_softmax = torch.nn.Softmax(dim = 1) 
        apply_sigmoid = torch.nn.Sigmoid()
        
        # create an empty tensor to store the hidden vector at each timestep
        Hs = torch.empty(X.shape[0], h0_2.shape[1], dtype = torch.float64)
        
        loss_list = []
        ct1 = self.ct_prev1.detach()
        ct2 = self.ct_prev2.detach()
        for t in range(self.tau):
            # input gate
            it1 = apply_sigmoid(torch.matmul(X[t:t+1,:], torch_network['Wix1']) + torch.matmul(hprev1, torch_network['Wih1']) + torch_network['bi1'])

            # candidate input
            c_tilde1 = apply_tanh(torch.matmul(X[t:t+1,:], torch_network['Wcx1']) + torch.matmul(hprev1, torch_network['Wch1']) + torch_network['bc1'])

            # forget gate
            ft2 = apply_sigmoid(torch.matmul(X[t:t+1,:], torch_network['Wfx1']) + torch.matmul(hprev1, torch_network['Wfh1']) + torch_network['bf1'])

            # update cell state
            ct1 = ft2*ct1 + it1*c_tilde1

            # output gate
            ot1 = apply_sigmoid(torch.matmul(X[t:t+1,:], torch_network['Wox1']) + torch.matmul(hprev1, torch_network['Woh1']) + torch_network['bo1'])

            # update hidden state (long-term memory)
            ht1 = ot1*apply_tanh(ct1)
            hprev1 = ht1

            x2 = ht1
            # input gate
            it2 = apply_sigmoid(torch.matmul(x2, torch_network['Wix2']) + torch.matmul(hprev2, torch_network['Wih2']) + torch_network['bi2'])

            # candidate input
            c_tilde2 = apply_tanh(torch.matmul(x2, torch_network['Wcx2']) + torch.matmul(hprev2, torch_network['Wch2']) + torch_network['bc2'])

            # forget gate
            ft2 = apply_sigmoid(torch.matmul(x2, torch_network['Wfx2']) + torch.matmul(hprev2, torch_network['Wfh2']) + torch_network['bf2'])

            # update cell state
            ct2 = ft2*ct2 + it2*c_tilde2

            # output gate
            ot2 = apply_sigmoid(torch.matmul(x2, torch_network['Wox2']) + torch.matmul(hprev2, torch_network['Woh2']) + torch_network['bo2'])
            ht2 = ot2*apply_tanh(ct2)
            hprev2 = ht2

            Hs[t:t+1, :] = ht2

        self.last_h1 = hprev1
        self.last_h2 = hprev2 
        self.ct_prev1 = ct1
        self.ct_prev2 = ct2       
        
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
    

    def training(self, X:list, y:list, Xval:np.ndarray, yval:np.ndarray, epochs:int, model_path = None, plot_loss = False):
        # Initialise lists etc
        loss_list = []
        val_loss = []
        t = 1
        print('Starting Training')
        start_time = perf_counter()
        for i in range(epochs): 
            print(f' ----- Epoch: {i+1} ------ ')
            # Shuffle data
            data = list(zip(X, y))
            self.rng.shuffle(data) 

            # Reset h between epochs
            self.last_h1 = torch.zeros(1, self.m1, dtype = torch.float64)
            self.last_h2 = torch.zeros(1, self.m2, dtype = torch.float64)

            # Iterate over articles
            for X_article, y_article in data:
                 # Requences from one article
                sequences = list(zip(X_article, y_article))
                
                for Xbatch, ybatch in sequences:
                    self.tau = Xbatch.shape[0]
                    ht1 = self.last_h1.detach().numpy()
                    ht2 = self.last_h2.detach().numpy()

                    # Forward- and backward pass
                    grads, loss = self.BackwardsPass(Xbatch, ybatch, ht1, ht2)


                    # Save loss
                    if t == 1:
                        smooth_loss = loss
                        loss_list.append(smooth_loss)
                    elif t % 10 == 0:
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
                    if t % 10000 == 0:
                        print(f'iteration: {t}')
                        val_loss.append(self.ComputeLoss(Xval, yval))
                    
                    t += 1 # increment iterations
        # Training time
        end_time = perf_counter()
        self.training_time = end_time - start_time
        print(f'Training took {round(self.training_time, 4)} seconds to execute')
        
        # Plot losses
        #self.plot_loss(t, loss_list, val_loss, model_path = model_path)
        return self.training_time, val_loss[-1], loss_list[-1]

    def plot_loss(self, t:int, smooth_loss:list, val_loss:list, model_path = None)->None:
        """
        plot_costs plots the cost, loss and accuracy for training and validation over the number of update steps\n
        
        :param smooth_loss: smooth loss values during training
        :type smooth_loss: list
        """
        # Plot Smooth Loss
        f_size = 25
        l_width = 3.0

        t_points = np.linspace(0, t-1, len(smooth_loss))

        plt.figure('Training Loss', figsize = (10,5))
        plt.plot(t_points, np.asarray(smooth_loss), 'b', label = 'Smooth Loss', linewidth = l_width)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xlabel('Update steps', fontsize = f_size)
        plt.ylabel('Smooth loss', fontsize = f_size)
        plt.xlim(0, t)
        plt.ylim(bottom = 0)
        plt.legend(fontsize = f_size)
        if model_path:
            filename = f"{model_path}/train_loss"
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()

        # Plot validation loss
        if len(val_loss) > 0:
            t_points = np.linspace(0, t-1, len(val_loss))
            f_size = 25
            l_width = 3.0

            plt.figure('Valiation Loss', figsize = (10,5))
            plt.plot(t_points, np.asarray(val_loss), 'b', label='Smooth Loss', linewidth=l_width)
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.xlabel('Update steps', fontsize = f_size)
            plt.ylabel('Smooth loss', fontsize = f_size)
            plt.xlim(0, t)
            plt.ylim(bottom = 0)
            plt.legend(fontsize = f_size)
            if model_path:
                filename = f"{model_path}/val_loss"
                plt.savefig(filename, bbox_inches = 'tight')
            else:
                plt.show()


    def save_model(self, model_path):
        model_data = {
            'self': self
        }
        filename = f"{model_path}/model"
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path):
        with open(f"{model_path}/model", 'rb') as f:
            model_data = pickle.load(f)
        self.lstm = model_data['lstm']
        self.char_to_ind = model_data['char_to_ind']
        self.ind_to_char = model_data['ind_to_char']
        self.m = model_data['m']
        self.K = model_data['K']
        self.eta = model_data['eta']


    def synthesize_text(self, x0:np.ndarray, text_length:int, test_loss = None, T = None, theta = None) -> str:
        chars = []

        # Load net
        torch_network = {}
        for kk in self.lstm.keys():
            torch_network[kk] = torch.tensor(self.lstm[kk], dtype = torch.float64, requires_grad=True)
     
        apply_tanh = torch.nn.Tanh()
        apply_sigmoid = torch.nn.Sigmoid()

        hprev1, hprev2 = self.last_h1.detach(), self.last_h2.detach()
        xt = torch.from_numpy(x0)
        ct1 = self.ct_prev1.detach()
        ct2 = self.ct_prev2.detach()
        for t in range(text_length):
            # input gate
            it1 = apply_sigmoid(torch.matmul(xt, torch_network['Wix1']) + torch.matmul(hprev1, torch_network['Wih1']) + torch_network['bi1'])

            # candidate input
            c_tilde1 = apply_tanh(torch.matmul(xt, torch_network['Wcx1']) + torch.matmul(hprev1, torch_network['Wch1']) + torch_network['bc1'])

            # forget gate
            ft2 = apply_sigmoid(torch.matmul(xt, torch_network['Wfx1']) + torch.matmul(hprev1, torch_network['Wfh1']) + torch_network['bf1'])

            # update cell state
            ct1 = ft2*ct1 + it1*c_tilde1

            # output gate
            ot1 = apply_sigmoid(torch.matmul(xt, torch_network['Wox1']) + torch.matmul(hprev1, torch_network['Woh1']) + torch_network['bo1'])

            # update hidden state (long-term memory)
            ht1 = ot1*apply_tanh(ct1)
            hprev1 = ht1

            x2 = ht1
            # input gate
            it2 = apply_sigmoid(torch.matmul(x2, torch_network['Wix2']) + torch.matmul(hprev2, torch_network['Wih2']) + torch_network['bi2'])

            # candidate input
            c_tilde2 = apply_tanh(torch.matmul(x2, torch_network['Wcx2']) + torch.matmul(hprev2, torch_network['Wch2']) + torch_network['bc2'])

            # forget gate
            ft2 = apply_sigmoid(torch.matmul(x2, torch_network['Wfx2']) + torch.matmul(hprev2, torch_network['Wfh2']) + torch_network['bf2'])

            # update cell state
            ct2 = ft2*ct2 + it2*c_tilde2

            # output gate
            ot2 = apply_sigmoid(torch.matmul(x2, torch_network['Wox2']) + torch.matmul(hprev2, torch_network['Woh2']) + torch_network['bo2'])
            ht2 = ot2*apply_tanh(ct2)
            hprev2 = ht2

            # new predictions 
            o_s = torch.matmul(ht2, torch_network['V']) + torch_network['c']
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
        if test_loss:
            text_seq += f'\n \n \n \n Test Loss: {test_loss} \n Training took {self.training_time:.2f} seconds'   
        else:
            text_seq += f'\n \n \n \n Training took {self.training_time:.2f} seconds'    
    
        return text_seq
        


def main():
    datamanager = DataManager()
    datamanager.read_files()
    ind_to_char, char_to_ind = datamanager.encode_data()
    
    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(42).state
    
    # Paramaters: ------------------- CHANGE HERE ---------------------------
    seq_length = 50
    m1, m2 = 200, 150    
    epochs = 10
    model_path = f'LSTM2/m1-{m1}_m2-{m2}_SL{seq_length}_epochs{epochs}/'
    os.makedirs(os.path.dirname(model_path), exist_ok = True)

    # Initialise LSTM
    lstm = LSTM2(m1 = m1, m2=m2, K = datamanager.K, eta=0.001, rng = rng, tau = seq_length, ind_to_char = ind_to_char, char_to_ind = char_to_ind)
    
    # Divide data in to sequences
    X_train, y_train = datamanager.create_article_sequences(datamanager.training_data, seq_length)
    X_val, y_val = datamanager.create_article_sequences(datamanager.validation_data, seq_length)
    X_test, y_test = datamanager.create_article_sequences(datamanager.test_data, seq_length)
    print('Sequences created')

   # X_train, y_train, X_val, y_val, X_test, y_test = X_train[0:1], y_train[0:1], X_val[0:10], y_val[0:10], X_test[0:10], y_test[0:10]
    
    # Train network
    lstm.training(X_train, y_train, X_val, y_val, epochs = epochs, model_path = model_path)
    
    # Compute test loss
    test_loss = lstm.ComputeLoss(X_test, y_test)
    print(f'test loss: {round(test_loss, 2)}')

    # Synthesize text
    # Generate starting character
    x0 = np.zeros((1, lstm.K), dtype = np.float64)
    ii = lstm.char_to_ind['T']
    x0[0, ii] = 1
    text_seq = lstm.synthesize_text(x0 = x0, text_length = 1000, test_loss = test_loss, T=None, theta=None)
    with open(f'{model_path}/text.txt', 'w') as f:
        f.write(text_seq)

    lstm.save_model(model_path = model_path)


if __name__== "__main__":
    main()