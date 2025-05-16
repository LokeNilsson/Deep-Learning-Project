import pickle
from PyDictionary import PyDictionary as pydict
import numpy as np
import random
import torch

class ModelEvaluator():
    def __init__(self):
        self.eng_dict = pydict()
        
        # Randomiser
        rng = np.random.default_rng()
        BitGen = type(rng.bit_generator)
        rng.bit_generator.state = BitGen(42).state
        self.rng = rng


    def load_net(self, model_path):
        with open(f"{model_path}/model", 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['self']


    def spellcheck(self, text: str):
        # Filter for words
        list_of_words = []
        word = ''
        
        for ch in text:
            if ch.isalpha():
                word += ch
            else:
                if len(word) > 1:
                    list_of_words.append(word)
                    word = ''
                else:
                    word = ''

        # Number of correctly spelled words
        correct_counter = 0
        for word in list_of_words:
            if word in self.eng_dict.keys():
                correct_counter += 1

        # Total Amount of words
        tot_words = len(list_of_words)

        spelling_accuracy = correct_counter / tot_words
        
        return tot_words, spelling_accuracy

def main():
    # Paramteres to fill in:
    model_path = 'LSTM1/m100_SL25_epochs10'
    text_length = 1000

    # Initialise Evaluator
    evaluator = ModelEvaluator()
    evaluator.load_net(model_path = model_path)   
    
    # Random starting character
    all_char_indices = list(evaluator.model.char_to_ind.values())
    start_char_idx = random.choice(all_char_indices)
    x0 = np.zeros((1, evaluator.K))
    x0[0, start_char_idx] = 1
    
    # Generate Text
    generated_text = evaluator.model.synthesize_text(x0 = x0, text_length = text_length, T = None, theta = None)
    print(generated_text)


    # Check accuracy
    # total_words, spelling_accuracy = evaluator.spellcheck(generated_text)
    # print(f'{model_type} model generated {total_words} with an accuracy of {spelling_accuracy:.2f} %')

    



if __name__== "__main__":
    main()