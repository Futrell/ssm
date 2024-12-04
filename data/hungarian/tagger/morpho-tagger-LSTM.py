import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()
        self.input_characters.add(' ')
        self.target_characters.add(' ')
        self.process_data()
        
    def process_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(num_samples, len(lines) - 1)]:
            parts = line.split('\t')
            if len(parts) < 5:
                continue  # Skip lines that don't have enough data
            _,input_text, analysis, segmentation, _ = parts
            target_text = '\t' + segmentation + '\t' + analysis + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                self.input_characters.add(char)
            for char in target_text:
                self.target_characters.add(char)
        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))
        self.num_encoder_tokens = len(self.input_characters) + 1
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

class Seq2SeqTrainer:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.input_token_index = dict([(char, i) for i, char in enumerate(data_processor.input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(data_processor.target_characters)])
        self.encoder_input_data = np.zeros((len(data_processor.input_texts), data_processor.max_encoder_seq_length, data_processor.num_encoder_tokens), dtype='float32')
        self.decoder_input_data = np.zeros((len(data_processor.input_texts), data_processor.max_decoder_seq_length, data_processor.num_decoder_tokens), dtype='float32')
        self.decoder_target_data = np.zeros((len(data_processor.input_texts), data_processor.max_decoder_seq_length, data_processor.num_decoder_tokens), dtype='float32')
        self.prepare_data()
        
    def prepare_data(self):
        for i, (input_text, target_text) in enumerate(zip(self.data_processor.input_texts, self.data_processor.target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.
            self.encoder_input_data[i, t + 1:, self.input_token_index[' ']] = 1.
            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
            self.decoder_input_data[i, t + 1:, self.target_token_index[' ']] = 1.
            self.decoder_target_data[i, t:, self.target_token_index[' ']] = 1.
    
    def build_model(self):
        encoder_inputs = Input(shape=(None, self.data_processor.num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        decoder_inputs = Input(shape=(None, self.data_processor.num_decoder_tokens))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.data_processor.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        
        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        
    def train(self, batch_size, epochs):
        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.2)
        
    def save_models(self, model_path, encoder_model_path, decoder_model_path):
        self.model.save(model_path.replace('.h5', '.keras'))
        self.encoder_model.save(encoder_model_path.replace('.h5', '.keras'))
        self.decoder_model.save(decoder_model_path.replace('.h5', '.keras'))

class Seq2SeqTester:
    def __init__(self, encoder_model_path, decoder_model_path, data_processor):
        self.encoder_model = load_model(encoder_model_path)
        self.decoder_model = load_model(decoder_model_path)
        self.data_processor = data_processor
        self.input_token_index = dict([(char, i) for i, char in enumerate(data_processor.input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(data_processor.target_characters)])
        self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())
        
        # Add an unknown token to handle characters not seen in training data
        self.unknown_token = "<UNK>"
        self.input_token_index[self.unknown_token] = data_processor.num_encoder_tokens - 1
        self.target_token_index[self.unknown_token] = len(self.target_token_index)

    def decode_sequences_in_batch(self, input_seqs):
        # Assuming input_seqs is a batch of sequences
        states_value = self.encoder_model.predict(input_seqs, verbose=0)  # Set verbose to 0 here
        target_seqs = np.zeros((len(input_seqs), 1, self.data_processor.num_decoder_tokens))
        target_seqs[:, 0, self.target_token_index['\t']] = 1.
        
        stop_conditions = [False] * len(input_seqs)
        decoded_sentences = [''] * len(input_seqs)
        while not all(stop_conditions):
            output_tokens, h, c = self.decoder_model.predict([target_seqs] + states_value, verbose=0)  # And here
        
            sampled_token_indices = np.argmax(output_tokens[:, -1, :], axis=1)
            
            for i, (sampled_token_index, decoded_sentence) in enumerate(zip(sampled_token_indices, decoded_sentences)):
                if not stop_conditions[i]:
                    sampled_char = self.reverse_target_char_index[sampled_token_index]
                    decoded_sentences[i] += sampled_char
                    
                    if sampled_char == '\n' or len(decoded_sentence) > self.data_processor.max_decoder_seq_length:
                        stop_conditions[i] = True

            target_seqs = np.zeros((len(input_seqs), 1, self.data_processor.num_decoder_tokens))
            for i, sampled_token_index in enumerate(sampled_token_indices):
                target_seqs[i, 0, sampled_token_index] = 1.
            
            states_value = [h, c]
        
        return decoded_sentences


    def word_to_encoder_input(self, word):
        encoder_input = np.zeros((1, self.data_processor.max_encoder_seq_length, self.data_processor.num_encoder_tokens), dtype='float32')
        for t, char in enumerate(word):
            if t >= self.data_processor.max_encoder_seq_length:  # Add this check
                break
            char_index = self.input_token_index.get(char, self.input_token_index[self.unknown_token])
            encoder_input[0, t, char_index] = 1.
        # Pad the rest of the sequence if necessary
        encoder_input[0, t:, self.input_token_index[' ']] = 1.
        return encoder_input


    def extract_analysis_segmentation(self, decoded_sentence):
        parts = decoded_sentence.strip().split('\t')
        if len(parts) == 3:
            segmentation, analysis = parts[1], parts[2]
        else:
            segmentation, analysis = '', ''
        return segmentation, analysis


    def test_on_csv(self, test_data_path, output_csv_path, batch_size=64):
        test_df = pd.read_csv(test_data_path, delimiter = ",")
        output_data = []
        input_seqs = []

        for index, row in test_df.iterrows():
            word = row['Word'].lower()
            input_seq = self.word_to_encoder_input(word)
            input_seqs.append(input_seq)

            # When batch_size is reached or it's the last word, predict in batch
            if len(input_seqs) == batch_size or index == len(test_df) - 1:
                batch_input_seqs = np.vstack(input_seqs)  # Stack input sequences
                decoded_sentences = self.decode_sequences_in_batch(batch_input_seqs)

                for decoded_sentence in decoded_sentences:
                    segmentation, analysis = self.extract_analysis_segmentation(decoded_sentence)
                    lemma = analysis.split(' ')[0] if analysis else ''
                    output_data.append({
                        "Word": word,
                        "Frequency": row['Frequency'],
                        "Lemma": lemma,
                        "Analysis": analysis,
                        "Segmentation": segmentation
                    })


                # Clear the input_seqs for the next batch
                input_seqs = []

            # Print status after every 1000 words
            if (index + 1) % 1000 == 0:
                print(f'Processed {index + 1} words')

        # Write to CSV in one go
        output_df = pd.DataFrame(output_data)
        output_df.sort_values(by='Frequency', ascending=False, inplace=True)
        output_df.to_csv(output_csv_path, index=False)



# Constants
data_path = 'data/hungarian/hungarian_unimorph.txt'
num_samples = 10000  # Number of samples to train on
latent_dim = 128  # Latent dimensionality of the encoding space
batch_size = 64  # Batch size for training
epochs = 2  # Number of epochs to train for

# Data Processing
data_processor = DataProcessor(data_path)

# Training
trainer = Seq2SeqTrainer(data_processor)
trainer.build_model()
trainer.train(batch_size, epochs)
trainer.save_models('data/hungarian/seq2seq/seq2seq_model.keras', 'data/hungarian/seq2seq/encoder_model.keras', 'data/hungarian/seq2seq/decoder_model.keras')

# Testing
tester = Seq2SeqTester('data/hungarian/seq2seq/encoder_model.keras', 'data/hungarian/seq2seq/decoder_model.keras', data_processor)
tester.test_on_csv('data/hungarian/CHILDES_ADULT_production.csv', 'data/hungarian/CHILDES_ADULT_production_seq2seq.csv')
