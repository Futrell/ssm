import torch
import csv
import pprint as pp
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

from collections import defaultdict

def load(file_path, col_separator, char_separator):
    wordlist = defaultdict(list)
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=col_separator)
        next(reader)
        for row in reader:
            # Ensure the row is not empty
            if row:
                row_val = True if len(row) == 1 or row[1] in ['TRUE', 'grammatical'] else False
                # If char_separator is None, split word character by character
                if not char_separator:
                    wordlist[row_val].append(list(row[0]))
                else:
                    # Split the first column (word) using the char_separator
                    wordlist[row_val].append(row[0].split(char_separator))

    return wordlist

def build_phone2ix(wordlist):
    # Build a mapping from symbols to integer indices
    wordlist = wordlist[True] + wordlist[False]
    unique_phones = sorted(set(symbol for word in wordlist for symbol in word))
    phone2ix = {phone: idx for idx, phone in enumerate(unique_phones)}
    return phone2ix

def wordlist_to_vec(wordlist, phone2ix):
    good = [torch.LongTensor(list(map(phone2ix.get, word))).to(DEVICE) for word in wordlist[True]]
    bad = [torch.LongTensor(list(map(phone2ix.get, word))).to(DEVICE) for word in wordlist[False]]
    return {True: good, False: bad}

def process_data(file_path, col_separator=",", char_separator=" "):
    wordlist = load(file_path, col_separator, char_separator)
    phone2ix = build_phone2ix(wordlist)

    word_vec = wordlist_to_vec(wordlist, phone2ix)
    return wordlist, phone2ix, word_vec
