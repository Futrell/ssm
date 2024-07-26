import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# Define the parameters and hyperparameters
batch_size = 64  # Batch size for training
epochs = 2      # Number of epochs to train for
latent_dim = 256 # Latent dimensionality of the encoding space
num_samples = 10000  # Number of samples to train on
max_length = 30  # Max sequence length for BPE tokens

# Initialize a tokenizer with BPE
tokenizer = Tokenizer(BPE()) #unk_token="[UNK]"
tokenizer.pre_tokenizer = Whitespace()

# Create a trainer for the tokenizer
trainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])

# Path to the data txt file on disk
data_path = 'data/hungarian/hungarian_unimorph.txt'

input_texts = []
segmentations = []

# Read the dataset and prepare text for the tokenizer
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[: min(num_samples, len(lines) - 1)]:
    parts = line.split('\t')
    if len(parts) < 5:
        continue  # Skip lines that don't have enough data
    input_text, target_text = parts[1], parts[3]  # Adjust according to your data format
    input_texts.append(input_text)
    
    # Encode the input text using the BPE tokenizer
    encoded = tokenizer.encode(input_text)
    segmentation = [0] * len(encoded.tokens)
    
    target_index = 0
    for token_index, token in enumerate(encoded.tokens):
        if target_text[target_index:target_index+len(token)].strip('-') == token:
            # If the token in target text matches the BPE token, move the target index
            target_index += len(token)
            # Handle possible dash '-' in the target text that indicates a split
            if target_index < len(target_text) and target_text[target_index] == '-':
                target_index += 1  # Skip the dash
        else:
            # If the token does not match, mark this as a split
            segmentation[token_index] = 1
            target_index += len(token)
    
    # Debug prints
    print(f"Input text: {input_text}")
    print(f"Encoded tokens: {encoded.tokens}")
    print(f"Target text: {target_text}")
    print(f"Segmentation: {segmentation}")
    
    segmentations.append(segmentation)
breakpoint()

# Train the tokenizer on the input texts
tokenizer.train_from_iterator(input_texts, trainer)

# Tokenize and pad sequences
input_sequences = [tokenizer.encode(text).ids for text in input_texts]
input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='post')

# Pad segmentation sequences and ensure they align with the tokenized input
segmentation_sequences = pad_sequences(segmentations, maxlen=max_length, padding='post')

# Convert the segmentations to categorical as we will be using softmax for predictions
segmentation_sequences = to_categorical(segmentation_sequences, num_classes=2)

# Define an input sequence and process it
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=tokenizer.get_vocab_size(), output_dim=latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Set up the decoder
decoder_inputs = Input(shape=(None, 2))  # There are 2 possible classes for each position: split or not split
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(2, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn encoder_input_data & decoder_input_data into decoder_target_data
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([input_sequences, segmentation_sequences], segmentation_sequences,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to something readable
reverse_input_char_index = dict(
    (i, char) for char, i in tokenizer.get_vocab().items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1 with a start character.
    target_seq = np.zeros((1, 1, 2))
    target_seq[0, 0, 0] = 1  # Start with "no split" class

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token and add the corresponding character to the decoded sentence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_input_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character.
        if (sampled_char == '[SEP]' or
           len(decoded_sentence) > max_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 2))
        target_seq[0, 0, sampled_token_index] = 1

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Test the decoding sequence
for seq_index in range(10):
    # Take one sequence for trying out decoding
    input_seq = input_sequences[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)



# Test the model with a new sequence
test_sentence = "malackodjunk"
test_seq = tokenizer.encode(test_sentence).ids
test_seq_padded = pad_sequences([test_seq], maxlen=max_length, padding='post')
print(decode_sequence(test_seq_padded))
