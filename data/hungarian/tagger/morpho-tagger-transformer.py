import numpy as np
import pandas as pd
import torch
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from collections import Counter
from tqdm import tqdm
import csv
import time
import math

class DataProcess(Dataset):
    def __init__(self, data, vocab, max_length=128, is_test=False):
        self.max_length = max_length
        self.data = data
        self.vocab = vocab
        self.is_test = is_test

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if not self.is_test:
            sf = row['SF']
            target_sequence = row['Segmentation'] + ' ' + row['Analysis']
            return torch.tensor(self.encode(sf)), torch.tensor(self.encode(target_sequence))
        else:
            sf = row['Word']
            return torch.tensor(self.encode(sf)), idx  # Return index for pairing prediction

    def __len__(self):
        return len(self.data)

    @staticmethod
    def build_vocab(data):
        counter = Counter()
        for idx in range(len(data)):
            row = data.iloc[idx]
            sf = row['SF'] if 'SF' in row else row['Word']
            target_sequence = row['Segmentation'] + ' ' + row['Analysis']
            counter.update(sf)
            counter.update(target_sequence)
        return {char: i + 4 for i, char in enumerate(counter)}

    def encode(self, sequence):
        return [self.vocab.get(char, 3) for char in sequence][:self.max_length]  # 3 is the token for unknown characters

    def decode(self, sequence):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join([inv_vocab.get(token, '') for token in sequence])

def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Use the imported TransformerEncoderLayer and TransformerDecoderLayer here
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.out = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        device = src.device  # Define device here

        # Generate source mask if not already created
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            self.src_mask = self._generate_square_subsequent_mask(len(src)).to(device)

        # Generate target mask
        tgt_mask = self._generate_square_subsequent_mask(len(tgt)).to(device)

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, self.src_mask)

        tgt = self.decoder(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask)  # Use tgt_mask here

        output = self.out(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def custom_collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=1, batch_first=True)  # 1 is the token for padding
    tgt_batch = pad_sequence(tgt_batch, padding_value=1, batch_first=True)  # 1 is the token for padding
    return src_batch.transpose(0, 1), tgt_batch.transpose(0, 1)  # Transpose for sequence length first


def train_epoch(model, optimizer, criterion, dataloader, device, vocab_size):
    model.train()
    total_loss = 0.
    for batch, (src, tgt) in enumerate(tqdm(dataloader, desc='Training')):
        # Check for out-of-range indices
        if src.max() > vocab_size or tgt.max() > vocab_size:
            print("Found token index out of range!")
            print(f"Max src index: {src.max()}, Max tgt index: {tgt.max()}")
            breakpoint()

        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:-1, :])
        loss = criterion(output.view(-1, output.size(-1)), tgt[1:, :].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch, (src, tgt) in enumerate(tqdm(dataloader, desc='Evaluating')):
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:-1, :])
            loss = criterion(output.view(-1, output.size(-1)), tgt[1:, :].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def test(model, dataloader, device, dataset, input_path, output_path):
    model.eval()
    results = []
    with torch.no_grad():
        for batch, (src, _) in enumerate(tqdm(dataloader, desc='Testing')):
            src = src.to(device)
            memory = model.transformer_encoder(model.encoder(src) * math.sqrt(model.d_model))
            tgt = torch.ones(1, src.size(1)).fill_(2).type_as(src)  # 2 is the token for start of sequence
            for i in range(128):
                output = model.out(model.transformer_decoder(model.decoder(tgt) * math.sqrt(model.d_model), memory))
                output = output.argmax(dim=2)[-1, :]
                tgt = torch.cat([tgt, output.unsqueeze(0)], dim=0)
                if (output == 3).all():  # 3 is the token for end of sequence
                    break
            src = src.to('cpu')
            tgt = tgt.to('cpu')
            for i in range(src.size(1)):
                input_sequence = dataset.decode(src[:, i].tolist())
                output_sequence = dataset.decode(tgt[:, i].tolist())
                results.append((input_sequence, output_sequence))

    # Load the input data to match words with their predicted segmentation and analysis
    input_data = pd.read_csv(input_path)
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Lemma', 'SF', 'Analysis', 'Segmentation', 'SF_Frequency'])
        for idx, row in input_data.iterrows():
            word = row['Word']
            freq = row['Frequency']
            try:
                # Find the prediction for the given word
                prediction = next(pred for input_seq, pred in results if input_seq == word)
                segmentation, analysis = prediction.split(' ')
                csvwriter.writerow([word, word, analysis, segmentation, freq])
            except StopIteration:
                # If no prediction, report and continue
                csvwriter.writerow([word, word, 'KeyError', 'KeyError', freq])


# Parameters for the model and training
d_model = 256  # Reduced from 512
nhead = 8
dim_feedforward = 1024  # Reduced from 2048
num_layers = 4  # Reduced from 6
dropout = 0.1
epochs = 20
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare the dataset and dataloader
train_data_path = 'data/hungarian/hungarian_unimorph.txt'
train_data = pd.read_csv(train_data_path, delimiter="\t")  # Pandas will automatically detect the delimiter

proportion_of_dataset = 0.01  # 1% of the dataset size
min_batch_size = 32  # minimum batch size
max_batch_size = 128  # maximum batch size that your hardware can handle efficiently

# Calculate batch size as a proportion of the dataset size
calculated_batch_size = int(len(train_data) * proportion_of_dataset)

# Ensure the batch size is within the specified bounds
batch_size = max(min(calculated_batch_size, max_batch_size), min_batch_size)

train_vocab = DataProcess.build_vocab(train_data)
breakpoint()
train_dataset = DataProcess(train_data, train_vocab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Prepare test data
test_data_path = 'data/hungarian/CHILDES_ADULT_production.csv'
test_data = pd.read_csv(test_data_path, delimiter=",")
test_dataset = DataProcess(test_data, train_vocab, is_test=True)  # Note the 'is_test' flag
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss, and optimizer
vocab_size = len(train_vocab)
model = TransformerModel(vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loss = train_epoch(model, optimizer, criterion, train_loader, device, vocab_size)
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | '
          f'average training loss {train_loss:8.5f} |')
    print('-' * 89)


# Test the model
test_output_path = 'data/hungarian/CHILDES_ADULT_production_seq2seq.csv'
test(model, test_loader, device, test_dataset, test_data_path, test_output_path)
