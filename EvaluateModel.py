import pickle
import numpy as np
import random
from nltk.corpus import words as corpus_words
import evaluate
from data_processing import DataManager
from LSTM1 import LSTM1
from LSTM2 import LSTM2
from RNN import RNN


class ModelEvaluator():
    def __init__(self):        
        # Randomiser
        rng = np.random.default_rng()
        BitGen = type(rng.bit_generator)
        rng.bit_generator.state = BitGen(42).state
        self.rng = rng

        self.english_vocab = set(w.lower() for w in corpus_words.words())

    def load_net(self, model_path):
        with open(f"{model_path}/model", 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['self']

    def percent_valid_words(self, text):
        initial_tokens = text.split()
        tokens = []
        for token in initial_tokens:
            if not token[-1].isalpha() and len(token)>1:
                if token[-2].isalpha():
                    tokens.append(token[:-1])
                    continue
            tokens.append(token)
                
        print(f'Percentage of valid words: {round(100*(sum(1 for t in tokens if t.lower() in self.english_vocab) / len(tokens)), 2)}%')
    
    def bleu_score(self, preds, refs):
        bleu = evaluate.load("sacrebleu")
        results = bleu.compute(predictions=[preds], references=[refs]) 
        print(f'Bleu: {results}')

def main():
    # get training data
    datamanager = DataManager()
    datamanager.read_files()
    datamanager.encode_data()
    

    ref_training_text = datamanager.training_data

    # Paramteres to fill in:
    # model_path = 'LSTM2/m1-100_m2-50_SL25_epochs1_eta0.01'
    # model_path = 'LSTM1/m150_SL25_epochs1_eta0.01'
    model_path = 'RNN/m100_SL50_epochs1'
    text_length = 200

    # X_test, y_test = datamanager.create_sequences(datamanager.test_data, seq_length=50)

    # Initialise Evaluator
    evaluator = ModelEvaluator()
    evaluator.load_net(model_path = model_path)

    # test_loss = evaluator.model.ComputeLoss(X_test, y_test)
    # print(f'test loss: {round(test_loss, 2)}')

    
    # Random starting character
    # all_char_indices = list(evaluator.model.char_to_ind.values())
    # start_char_idx = random.choice(all_char_indices)
    # x0 = np.zeros((1, evaluator.model.K))
    # x0[0, start_char_idx] = 1

    # set starting character
    x0 = np.zeros((1, evaluator.model.K), dtype = np.float64)
    ii = evaluator.model.char_to_ind[' ']
    x0[0, ii] = 1
    
    # Generate Text
    generated_text = evaluator.model.synthesize_text(x0 = x0, text_length = text_length, T = None, theta = 0.6)
    print(generated_text)
    evaluator.bleu_score(generated_text, ref_training_text)
    evaluator.percent_valid_words(generated_text)


if __name__== "__main__":
    main()