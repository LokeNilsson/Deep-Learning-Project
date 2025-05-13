import torch 
from data_processing import DataManager
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import perf_counter

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


    def ComputeValidationLoss(self, X:list, y:list, h0_np:np.ndarray)->list:
        ht = torch.from_numpy(h0_np)
        
        tau = self.tau

        torch_network = {}
        for kk in self.rnn.keys():
            torch_network[kk] = torch.tensor(self.rnn[kk], requires_grad=True)

        # Create activation functions       
        apply_tanh = torch.nn.Tanh()
        apply_softmax = torch.nn.Softmax(dim=1) 
        
        # create an empty tensor to store the hidden vector at each timestep
        Hs = torch.empty(self.tau, h0_np.shape[1], dtype=torch.float64)
        
        loss_list = []
        hprev = ht
        for i, (Xbatch_np, ybatch) in enumerate(zip(X, y)):
            Xbatch = torch.from_numpy(Xbatch_np) 
            for t in range(self.tau):
                a = torch.matmul(hprev, torch_network['W']) + torch.matmul(Xbatch[t:t+1], torch_network['U']) + torch_network['b']
                assert a.shape == (1,self.m)

                ht = apply_tanh(a)
                Hs[t:t+1, :] = ht

                hprev = ht
            
            Os = torch.matmul(Hs, torch_network['V']) + torch_network['c']       
            P = apply_softmax(Os)    
            
            # compute the loss
            loss = torch.mean(-torch.log(P[np.arange(tau), ybatch]))
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
    

    def training(self, X:list, y:list, Xval:np.ndarray, yval:np.ndarray, epochs:int):
        loss_list = []
        val_loss = []
        t = 1
        start_time = perf_counter()
        for _ in range(epochs):
            #X, y = self.rng.shuffle(X), self.rng.shuffle(y)

            h0 = np.zeros((1, self.m))
            for Xbatch, ybatch in zip(X, y):
                grads, loss = self.BackwardsPass(Xbatch, ybatch, h0)
                if t == 1:
                    smooth_loss = loss
                    loss_list.append(smooth_loss)

                elif t % 10 == 0:
                    smooth_loss = 0.999 * smooth_loss + 0.001 *loss 
                    loss_list.append(smooth_loss)

                # SGD using Adam
                for kk in grads.keys():
                    self.m_adam[kk]      = self.beta1 * self.m_adam[kk] + (1-self.beta1)*grads[kk]
                    self.v_adam[kk]      = self.beta2 * self.v_adam[kk] + (1-self.beta2)*(grads[kk]**2)
                    self.m_hat_adam[kk]  = self.m_adam[kk]/(1-self.beta1**t)
                    self.v_hat_adam[kk]  = self.v_adam[kk]/(1-self.beta2**t)
                    self.rnn[kk]         = self.rnn[kk] - (self.eta/(np.sqrt(self.v_hat_adam[kk]) + self.eps))*self.m_hat_adam[kk]
                
                # Compute validation loss
                if t % 100 == 0:
                    print(f'iteration: {t}')
                    #val_loss.append(self.ComputeValidationLoss(Xval, yval, h0_np = np.zeros((1, self.m))))
                    
                # increment iterations
                t += 1
        end_time = perf_counter()
        print(f'Training took {round(end_time-start_time, 4)} seconds to execute')
        self.plot_loss(loss_list, val_loss)


    def plot_loss(self, smooth_loss:list, val_loss:list)->None:
        """
        plot_costs plots the cost, loss and accuracy for training and validation over the number of update steps\n
        
        :param smooth_loss: smooth loss values during training
        :type smooth_loss: list
        """

        # Plot Smooth Loss
        f_size = 25
        l_width = 3.0

        plt.figure('Training Loss', figsize = (10,5))
        plt.plot(np.asarray(smooth_loss), 'b', label = 'Smooth Loss', linewidth = l_width)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xlabel('Update steps', fontsize = f_size)
        plt.ylabel('Smooth loss', fontsize = f_size)
        # plt.xlim(0, len(smooth_loss))
        plt.ylim(bottom = 0)
        plt.legend(fontsize = f_size)
        plt.savefig('train_loss.png', bbox_inches='tight')


        # Plot validation loss
        if len(val_loss) > 0:
            f_size = 25
            l_width = 3.0

            plt.figure('Valiation Loss', figsize = (10,5))
            plt.plot(np.asarray(val_loss), 'b', label='Smooth Loss', linewidth=l_width)
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.xlabel('Update steps', fontsize = f_size)
            plt.ylabel('Smooth loss', fontsize = f_size)
            # plt.xlim(0, len(smooth_loss))
            plt.ylim(bottom = 0)
            plt.legend(fontsize = f_size)
            plt.savefig('val_loss.png', bbox_inches = 'tight')
    
    def save_model(self, model_name):
        model_data = {
            'RNN': self.rnn,
            'char_to_ind': self.char_to_ind,
            'ind_to_char': self.ind_to_char, 
            'm': self.m,
            'K': self.K,
            'eta': self.eta,
        }
        with open(f'models/{model_name}', 'wb') as f:
            pickle.dump(model_data, f)
    
    
    def load_model(self, model_name):
        with open(f'models/{model_name}', 'rb') as f:
            model_data = pickle.load(f)
        self.rnn = model_data['RNN']
        self.char_to_ind = model_data['char_to_ind']
        self.ind_to_char = model_data['ind_to_char']
        self.m = model_data['m']
        self.K = model_data['K']
        self.eta = model_data['eta']


    def synthesize_text(self, x0:np.ndarray, text_length:int, save = False, T = None, theta = None) -> str:
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

        if save:
            with open(f'text.txt', 'w') as f:
                f.write(text_seq)
        return text_seq
        
def main():
    datamanager = DataManager()
    datamanager.read_files()
    ind_to_char, char_to_ind = datamanager.encode_data()
    
    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(42).state

    seq_length = 25
    m = 10
    rnn = RNN(m = m, K = datamanager.K, eta=0.001, rng = rng ,tau = seq_length, ind_to_char = ind_to_char, char_to_ind = char_to_ind)
    
    # Divide data in to sequences
    X_train, y_train = datamanager.create_sequences(datamanager.training_data, seq_length)

    X_val, y_val = datamanager.create_sequences(datamanager.validation_data, seq_length)
    X_test, y_test = datamanager.create_sequences(datamanager.test_data, seq_length)

    print(len(X_train))

    # train network
    rnn.training(X_train, y_train, X_val, y_val, epochs=1)

    test_loss = rnn.ComputeValidationLoss(X_test, y_test, h0_np = np.zeros((1,m)))
    print(f'test loss: {round(test_loss, 2)}')

    rnn.synthesize_text(x0=X_test[0][0:1, :], text_length=100, save=True)
    rnn.save_model('test1')


if __name__== "__main__":
    main()