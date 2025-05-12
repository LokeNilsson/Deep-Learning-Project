import torch 
from data_processing import DataManager
import numpy as np
import matplotlib.pyplot as plt

class RNN:
    def __init__(self, m, K, rng, tau):
        # Networks shapes
        self.m = m
        self.K = K
        
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
        self.RNN = {
            'b' : b,
            'c' : c,
            'U' : U,
            'W' : W,
            'V' : V
        }

        self.m_adam      = {'b': torch.zeros_like(b), 'c' : torch.zeros_like(c), 'U' : torch.zeros_like(U), 'W' : torch.zeros_like(W), 'V' : torch.zeros_like(V)}
        self.v_adam      = {'b': torch.zeros_like(b), 'c' : torch.zeros_like(c), 'U' : torch.zeros_like(U), 'W' : torch.zeros_like(W), 'V' : torch.zeros_like(V)}
        self.m_hat_adam  = {'b': torch.zeros_like(b), 'c' : torch.zeros_like(c), 'U' : torch.zeros_like(U), 'W' : torch.zeros_like(W), 'V' : torch.zeros_like(V)}
        self.v_hat_adam  = {'b': torch.zeros_like(b), 'c' : torch.zeros_like(c), 'U' : torch.zeros_like(U), 'W' : torch.zeros_like(W), 'V' : torch.zeros_like(V)}
        self.beta1       = 0.9
        self.beta2       = 0.999
        self.eps         = 1e-8


    def ComputeValidationLoss(self, X:torch.tensor, y:torch.tensor, h0:torch.tensor)->list:
        torch_network = {}
        for kk in RNN.keys():
            torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)

        ## give informative names to these torch classes        
        apply_tanh = torch.nn.Tanh()
        apply_softmax = torch.nn.Softmax(dim=1) 
        
        # create an empty tensor to store the hidden vector at each timestep
        Hs = torch.empty(X.shape[0], h0.shape[1], dtype=torch.float64)
        
        loss_list = []
        hprev = ht
        for t in range(self.tau):
            a = torch.matmul(hprev, torch_network['W']) + torch.matmul(X[t:t+1,:], torch_network['U']) + torch_network['b']

            ht = apply_tanh(a)
            Hs[t:t+1, :] = ht

            hprev = ht
            
            Os = torch.matmul(Hs, torch_network['V']) + torch_network['c']       
            P = apply_softmax(Os)    
            
            # compute the loss
            loss = torch.mean(-torch.log(P[np.arange(self.tau), y]))
            loss_list.append(loss)
        
        return np.mean(loss_list)


    def BackwardsPass(self, X:torch.tensor, y:torch.tensor, h0:torch.tensor) -> dict:
        torch_network = {}
        for kk in RNN.keys():
            torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)

        ## give informative names to these torch classes        
        apply_tanh = torch.nn.Tanh()
        apply_softmax = torch.nn.Softmax(dim=1) 
        
        # create an empty tensor to store the hidden vector at each timestep
        Hs = torch.empty(X.shape[0], h0.shape[1], dtype=torch.float64)
        
        hprev = ht
        loss_list = []
        for t in range(self.tau):
            a = torch.matmul(hprev, torch_network['W']) + torch.matmul(X[t:t+1,:], torch_network['U']) + torch_network['b']

            ht = apply_tanh(a)
            Hs[t:t+1, :] = ht

            hprev = ht
            
            Os = torch.matmul(Hs, torch_network['V']) + torch_network['c']       
            P = apply_softmax(Os)    
            
            # compute the loss
            loss = torch.mean(-torch.log(P[np.arange(self.tau), y]))
            loss_list.append(loss)

            # compute the backward pass relative to the loss and the named parameters 
            loss.backward()

            # extract the computed gradients and make them numpy arrays
            grads = {}
            for kk in RNN.keys():
                grads[kk] = torch_network[kk]

            grads['loss'] = loss
            grads['H'] = Hs
            grads['P'] = P

        loss_mean = np.mean(loss_list)
        return grads, loss_mean
    

    def training(self, X:list, y:list, Xval:list, yval:list, epochs:int):
        
        loss_list = []
        val_loss = []
        t = 0
        for _ in range(epochs):
            X, y = self.rng.shuffle(X), self.rng.shuffle(y)
            h0 = torch.zeros(1, self.m)
            for Xbatch, y_batch in zip(X, y):
                grads, loss = self.BackwardsPass(Xbatch, y_batch, h0)
                loss_list.append(loss)
                # SGD using Adam
                for kk in self.grads.keys():
                    self.m_adam[kk]      = self.beta1 * self.m_adam[kk] + (1-self.beta1)*grads[kk]
                    self.v_adam[kk]      = self.beta2 * self.v_adam[kk] + (1-self.beta2)*(grads[kk]**2)
                    self.m_hat_adam[kk] = self.m_adam[kk]/(1-self.beta1**t)
                    self.v_hat_adam[kk]  = self.v_adam[kk]/(1-self.beta2**t)
                    RNN[kk] = RNN[kk] - (self.eta/(np.sqrt(self.v_hat_adam[kk]) + self.eps))*self.m_hat_adam[kk]
                
                # Compute validation loss
                if t % 100 == 0:
                    val_loss.append(self.ComputeValidationLoss(Xval, yval, h0 = torch.zeros(1, self.m)))

                # increment iterations
                t += 1

        
        self.plot_smooth_loss(loss_list)


    def plot_smooth_loss(self, smooth_loss:list)->None:
        """
        plot_costs plots the cost, loss and accuracy for training and validation over the number of update steps\n
        
        :param smooth_loss: smooth loss values during training
        :type smooth_loss: list
        """
        f_size = 25
        l_width = 3.0

        plt.figure('Loss', figsize=(10,5))
        plt.plot(np.asarray(smooth_loss), 'b', label='smooth loss', linewidth=l_width)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Update steps', fontsize=f_size)
        plt.ylabel('Smooth loss', fontsize=f_size)
        # plt.xlim(0, len(smooth_loss))
        plt.ylim(bottom=0)
        plt.legend(fontsize=f_size)
        plt.savefig('smooth_loss.png', bbox_inches='tight')


def main():
    datamanager = DataManager()
    datamanager.read_files()
    datamanager.encode_data()
    
    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(42).state

    seq_length = 25
    rnn = RNN(m=100, K=datamanager.K, rng=rng ,tau=seq_length)




if __name__== "__main__":
    main()