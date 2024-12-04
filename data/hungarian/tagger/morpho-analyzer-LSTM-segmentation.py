import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Define paths
data_path = 'data/hungarian/hungarian_unimorph.txt'

# Define parameters
batch_size = 64
epochs = 100.  
latent_dim = 256
num_samples = 10000

# Prepare the data
input_texts = []
target_texts = []
input_characters = set([' '])  # Start with space to pad
target_characters = set(['\t', '\n'])  # Start and end tokens
target_characters.add(' ')

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[:min(num_samples, len(lines) - 1)]:
    _, input_text, _, segmentation, _ = line.split('\t')[:5]
    target_text = '\t' + segmentation + '\n'  # Use '\t' as start and '\n' as end
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        input_characters.add(char)
    for char in target_text:
        target_characters.add(char)

# Sort characters
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

# Character to index conversion dictionaries
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# Data vectorization
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, len(input_characters)), dtype='float32'
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, len(target_characters)), dtype='float32'
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, len(target_characters)), dtype='float32'
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[' ']] = 1.0
    for t, char in enumerate(target_text):
        # Decoder_target_data is ahead by one timestep and will not include the start token.
        decoder_input_data[i, t, target_token_index[char]] = 1.0

        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[' ']] = 1.0
    decoder_target_data[i, t :, target_token_index[' ']] = 1.0

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, len(input_characters)))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, len(target_characters)))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(len(target_characters), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Save the model if you need to use it later
# model.save('morphological_parser.h5')

# Inference setup
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1 with only the start character.
    target_seq = np.zeros((1, 1, len(target_characters)))
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token and add the corresponding character to the decoded sentence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character.
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, len(target_characters)))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Test the model with a new word
def word_to_encoder_input(word):
    encoder_input_data = np.zeros((1, max_encoder_seq_length, len(input_characters)), dtype='float32')
    for t, char in enumerate(word):
        encoder_input_data[0, t, input_token_index[char]] = 1.0
    encoder_input_data[0, t + 1:, input_token_index[' ']] = 1.0  # Pad the rest of the sequence
    return encoder_input_data

def extract_analysis_segmentation(decoded_sentence):
    # Assuming the decoded sentence contains '+' as the segmentation marker
    segments = decoded_sentence.strip().split('+')
    segments = [seg.strip() for seg in segments if seg]  # Remove empty segments and strip spaces
    return segments

# Example of processing a new word
new_word = "malackodjunk"
encoder_input = word_to_encoder_input(new_word)
decoded_sentence = decode_sequence(encoder_input)
segments = extract_analysis_segmentation(decoded_sentence)
print(f"Segments for '{new_word}': {segments}")
