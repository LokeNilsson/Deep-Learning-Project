import torch 
from data_processing import DataManager
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import perf_counter
import os

class Transformer:
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
        Wq = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 
        Wk = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 
        Wv = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 

        # Self-Attention Layers
        Wq_y = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 
        Wk_y = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 
        Wv_y = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 

        # Decoder
        Wq_dec = (1/np.sqrt(2*K))*rng.standard_normal(size = (K, m), dtype=np.float64) 
        
        V_dec = (1/np.sqrt(m))*rng.standard_normal(size = (m, K), dtype=np.float64) 
        c_dec = np.zeros((1, K), dtype=np.float64)                
        
        # Linear output Layers
        Vx = (1/np.sqrt(m))*rng.standard_normal(size = (m, K), dtype=np.float64) 
        cx = np.zeros((1, K), dtype=np.float64)


        # store parameters in dictionary
        self.transformer = {
            'Wq' : Wq,
            'Wk' : Wk,
            'Wv' : Wv,
            ##
            'Wq_y' : Wq_y,
            'Wk_y' : Wk_y,
            'Wv_y' : Wv_y,
            ##
            'Vx' : Vx,
            'cx' : cx,
            #
            'Wq_dec' : Wq_dec,
            'V_dec' : V_dec,
            'c_dec' : c_dec
        }

        adam_params = {}
        adam_params['beta1'] = 0.9
        adam_params['beta2'] = 0.999
        adam_params['eps'] = 1e-8
        for kk in self.transformer.keys():
            adam_params[f'm_{kk}'] = np.zeros_like(self.transformer[kk])
            adam_params[f'mhat_{kk}'] = np.zeros_like(self.transformer[kk])
            adam_params[f'v_{kk}'] = np.zeros_like(self.transformer[kk])
            adam_params[f'vhat_{kk}'] = np.zeros_like(self.transformer[kk])
        self.adam_params = adam_params


    def ComputeLoss(self, X:list, y:list)->list:
        torch_network = {}
        for kk in self.transformer.keys():
            torch_network[kk] = torch.tensor(self.transformer[kk], requires_grad=True)

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
                apply_layernorm = torch.nn.LayerNorm(self.m, bias=False, dtype=torch.float64) 

                # ----- Encoder -----
                # calculate query, key and value matrices
                Qx = torch.matmul(X, torch_network['Wq'])
                Kx = torch.matmul(X, torch_network['Wk'])
                Vx = torch.matmul(X, torch_network['Wv'])

                # get attention weights and multiply with values
                attn_weights = apply_softmax((torch.matmul(Qx, Kx.T)/torch.sqrt(torch.tensor(self.m))))

                Zx = torch.matmul(attn_weights, Vx)

                # Add and normalization layer
                Zx = apply_layernorm(X+Zx)

                # Feedforward       
                FFx = torch.matmul(Zx, torch_network['Vx']) + torch_network['cx'] 

                # Add and normaliation layer
                FFx = apply_layernorm(Zx+FFx)   

                # Transform output to key and values used in decoder
                K_enc = torch.matmul(FFx, torch_network['Wk'])
                V_enc = torch.matmul(FFx, torch_network['Wv'])
                
                # ----- Decoder -----
                # self attention layer using the output from the encoder as input
                Qy = torch.matmul(FFx, torch_network['Wq_y'])
                Ky = torch.matmul(FFx, torch_network['Wk_y'])
                Vy = torch.matmul(FFx, torch_network['Wv_y'])

                # get attention weights and multiply with values 
                #  masking future positions (setting them to -inf) before the softmax step 
                scores = (torch.matmul(Qy, Ky.T)/torch.sqrt(torch.tensor(self.m)))
                upper_tri_inds = torch.triu_indices(scores.shape[0], scores.shape[1], offset=1)
                scores[upper_tri_inds[0], upper_tri_inds[1]] = -torch.inf
                attn_weightsy = apply_softmax(scores)
                Zy = torch.matmul(attn_weightsy, Vy)

                # Add and normalization layer
                Zy = apply_layernorm(FFx+Zy)

                # encoder-decoder attention
                Q_dec = torch.matmul(Zy, torch_network['Wq_dec'])

                scores = (torch.matmul(Q_dec, K_enc.T)/torch.sqrt(torch.tensor(self.m)))
                attn_weights_dec = apply_softmax(scores)
                Z_dec = torch.matmul(attn_weights_dec, V_enc)
                
                # Add and normalization layer
                Z_dec = apply_layernorm(Zy+Z_dec)

                # linear layer
                Os = torch.matmul(Z_dec, torch_network['V_dec']) + torch_network['c_dec'] 

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
        for kk in self.transformer.keys():
            torch_network[kk] = torch.tensor(self.transformer[kk], dtype = torch.float64, requires_grad=True)

        apply_softmax = torch.nn.Softmax(dim=1)
        apply_layernorm = torch.nn.LayerNorm(self.m, bias=False, dtype=torch.float64)

        # ----- Encoder -----
        # calculate query, key and value matrices
        Qx = torch.matmul(X, torch_network['Wq'])
        Kx = torch.matmul(X, torch_network['Wk'])
        Vx = torch.matmul(X, torch_network['Wv'])

        # get attention weights and multiply with values
        attn_weights = apply_softmax((torch.matmul(Qx, Kx.T)/torch.sqrt(torch.tensor(self.m))))

        Zx = torch.matmul(attn_weights, Vx)

        # Add and normalization layer
        Zx = apply_layernorm(X+Zx)

        # Feedforward       
        FFx = torch.matmul(Zx, torch_network['Vx']) + torch_network['cx'] 

        # Add and normaliation layer
        FFx = apply_layernorm(Zx+FFx)   

        # Transform output to key and values used in decoder
        K_enc = torch.matmul(FFx, torch_network['Wk'])
        V_enc = torch.matmul(FFx, torch_network['Wv'])
        
        # ----- Decoder -----
        # self attention layer using the output from the encoder as input
        Qy = torch.matmul(FFx, torch_network['Wq_y'])
        Ky = torch.matmul(FFx, torch_network['Wk_y'])
        Vy = torch.matmul(FFx, torch_network['Wv_y'])

        # get attention weights and multiply with values 
        #  masking future positions (setting them to -inf) before the softmax step 
        scores = (torch.matmul(Qy, Ky.T)/torch.sqrt(torch.tensor(self.m)))
        upper_tri_inds = torch.triu_indices(scores.shape[0], scores.shape[1], offset=1)
        scores[upper_tri_inds[0], upper_tri_inds[1]] = -torch.inf
        attn_weightsy = apply_softmax(scores)
        Zy = torch.matmul(attn_weightsy, Vy)

        # Add and normalization layer
        Zy = apply_layernorm(FFx+Zy)

        # encoder-decoder attention
        Q_dec = torch.matmul(Zy, torch_network['Wq_dec'])

        scores = (torch.matmul(Q_dec, K_enc.T)/torch.sqrt(torch.tensor(self.m)))
        attn_weights_dec = apply_softmax(scores)
        Z_dec = torch.matmul(attn_weights_dec, V_enc)
        
        # Add and normalization layer
        Z_dec = apply_layernorm(Zy+Z_dec)

        # linear layer
        Os = torch.matmul(Z_dec, torch_network['V_dec']) + torch_network['c_dec'] 

        # Softmax to get final predictions
        P = apply_softmax(Os)

        # compute the loss
        loss = torch.mean(-torch.log(P[np.arange(self.tau), y]))

        # compute the backward pass relative to the loss and the named parameters 
        loss.backward()

        # extract the computed gradients and make them numpy arrays
        grads = {}
        for kk in self.transformer.keys():
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
            
                        self.transformer[kk] = self.transformer[kk] - (self.eta/(np.sqrt(self.adam_params[f'vhat_{kk}']) + self.adam_params['eps']))*self.adam_params[f'mhat_{kk}']
                    
                    # Validation
                    if t % 10000 == 0:
                        print(f'iteration: {t}')
                        val_loss.append(self.ComputeLoss(Xval, yval))

                    t += 1
                    
        # Training time
        end_time = perf_counter()
        self.training_time = end_time - start_time
        print(f'Training took {round(self.training_time, 4)} seconds to execute')
        
        # Plot losses
        self.plot_loss(t, loss_list, val_loss, model_path = model_path)
        return self.transformer

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
        self.transformer = model_data['transformer']
        self.char_to_ind = model_data['char_to_ind']
        self.ind_to_char = model_data['ind_to_char']
        self.m = model_data['m']
        self.K = model_data['K']
        self.eta = model_data['eta']
        self.last_h = self.last_h


    def synthesize_text(self, x0:np.ndarray, text_length:int, test_loss = None, T = None, theta = None) -> str:
        chars = []
        apply_softmax = torch.nn.Softmax(dim=1)
        torch_network = {}
        for kk in self.transformer.keys():
            torch_network[kk] = torch.tensor(self.transformer[kk], dtype = torch.float64, requires_grad=True)

        xt = torch.from_numpy(x0)
        for t in range(text_length):
            apply_layernorm = torch.nn.LayerNorm(self.m, bias=False, dtype=torch.float64) 

            # ----- Encoder -----
            # calculate query, key and value matrices
            Qx = torch.matmul(xt, torch_network['Wq'])
            Kx = torch.matmul(xt, torch_network['Wk'])
            Vx = torch.matmul(xt, torch_network['Wv'])

            # get attention weights and multiply with values
            attn_weights = apply_softmax((torch.matmul(Qx, Kx.T)/torch.sqrt(torch.tensor(self.m))))

            Zx = torch.matmul(attn_weights, Vx)

            # Add and normalization layer
            Zx = apply_layernorm(xt+Zx)

            # Feedforward       
            FFx = torch.matmul(Zx, torch_network['Vx']) + torch_network['cx'] 

            # Add and normaliation layer
            FFx = apply_layernorm(Zx+FFx)   

            # Transform output to key and values used in decoder
            K_enc = torch.matmul(FFx, torch_network['Wk'])
            V_enc = torch.matmul(FFx, torch_network['Wv'])
            
            # ----- Decoder -----
            # self attention layer using the output from the encoder as input
            Qy = torch.matmul(FFx, torch_network['Wq_y'])
            Ky = torch.matmul(FFx, torch_network['Wk_y'])
            Vy = torch.matmul(FFx, torch_network['Wv_y'])

            # get attention weights and multiply with values 
            #  masking future positions (setting them to -inf) before the softmax step 
            scores = (torch.matmul(Qy, Ky.T)/torch.sqrt(torch.tensor(self.m)))
            upper_tri_inds = torch.triu_indices(scores.shape[0], scores.shape[1], offset=1)
            scores[upper_tri_inds[0], upper_tri_inds[1]] = -torch.inf
            attn_weightsy = apply_softmax(scores)
            Zy = torch.matmul(attn_weightsy, Vy)

            # Add and normalization layer
            Zy = apply_layernorm(FFx+Zy)

            # encoder-decoder attention
            Q_dec = torch.matmul(Zy, torch_network['Wq_dec'])

            scores = (torch.matmul(Q_dec, K_enc.T)/torch.sqrt(torch.tensor(self.m)))
            attn_weights_dec = apply_softmax(scores)
            Z_dec = torch.matmul(attn_weights_dec, V_enc)
            
            # Add and normalization layer
            Z_dec = apply_layernorm(Zy+Z_dec)

            # linear layer
            o_s = torch.matmul(Z_dec, torch_network['V_dec']) + torch_network['c_dec'] 

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
    epochs = 30
    model_path = f'transformer/m{m}_SL{seq_length}_epochs{epochs}/'
    os.makedirs(os.path.dirname(model_path), exist_ok = True)

    # Initialise transformer
    transformer = Transformer(m = m, K = datamanager.K, eta = 0.001, rng = rng, tau = seq_length, ind_to_char = ind_to_char, char_to_ind = char_to_ind)
    
    # Divide data in to sequences
    X_train, y_train = datamanager.create_article_sequences(datamanager.training_data, seq_length=seq_length)
    X_val, y_val = datamanager.create_article_sequences(datamanager.validation_data, seq_length=seq_length)
    X_test, y_test = datamanager.create_article_sequences(datamanager.test_data, seq_length=seq_length)
    print('Sequences created')

    X_train, y_train, X_val, y_val, X_test, y_test = X_train[0:1], y_train[0:1], X_val[0:10], y_val[0:10], X_test[0:10], y_test[0:10]
    
    # Train network
    transformer.training(X_train, y_train, X_val, y_val, epochs = epochs, model_path = model_path)
    
    # Compute test loss
    test_loss = transformer.ComputeLoss(X_test, y_test)
    print(f'test loss: {round(test_loss, 2)}')
    
    # Synthesize textproject/transformer.py

    # Generate starting letter
    x0 = np.zeros((1, transformer.K), dtype = np.float64)
    ii = transformer.char_to_ind['T']
    x0[0, ii] = 1
    text_seq = transformer.synthesize_text(x0 = x0, text_length = 1000, test_loss = test_loss)
    with open(f'{model_path}/text.txt', 'w') as f:
        f.write(text_seq)


    transformer.save_model(model_path = model_path)


if __name__== "__main__":
    main()