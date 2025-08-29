import torch
import csv
import pprint as pp
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from collections import defaultdict

def load(file_path, col_separator, char_separator, paired=False):
    wordlist = defaultdict(list)
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=col_separator)
        for row in reader:
            # Ensure the row is not empty
            if row:
                if not paired:
                    row_val = True if len(row) == 1 or row[1] == 'TRUE' else False

                    # If char_separator is None, split word character by character
                    if not char_separator:
                        wordlist[row_val].append(list(row[0]))
                    else:
                        # Split the first column (word) using the char_separator
                        wordlist[row_val].append(row[0].split(char_separator))
                else:
                    wordlist[True].append(row[0])
                    wordlist[False].append(row[1])
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

def pairing(input_data):
    """
    Purpose: Pairing the good and bad data
    """
    good = input_data[True]
    bad = input_data[False]
    pairs = zip(good, bad)
    pairs = [(x, y) for x, y in pairs]

    paired_data = {True: [x for x, y in pairs], False: [y for x, y in pairs]}
    return paired_data

# TODO: Need to tweak this to work with unpaired data
def process_data(file_path, col_separator=",", char_separator=" ", paired=False):
    wordlist = load(file_path, col_separator, char_separator, paired=paired)
    paired_wordlist = pairing(wordlist) if wordlist[False] else wordlist
    phone2ix = build_phone2ix(paired_wordlist)

    word_vec = wordlist_to_vec(paired_wordlist, phone2ix)
    return paired_wordlist, phone2ix, word_vec

# end def
if __name__ == "__main__":
    file_path  = 'data/mlregtest/04.04.SL.2.1.0_TestLR.txt'
    process_data(file_path, col_separator="\t", char_separator="")

    # dataset is too big---define a function that import a large dataset as zip.
    # option: use ssh fuse
