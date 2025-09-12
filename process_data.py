import torch
import csv
import pprint as pp
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ADD_EOS = True

from collections import defaultdict

def load_file(filepath, col_separator, char_separator, header=True, paired=False):
    wordlist = []
    with open(filepath, 'rt') as f:
        reader = csv.reader(f, delimiter=col_separator)
        header = next(reader) if header else None
        for row in reader:
            if row:
                if paired:
                    form1 = row[0].split(char_separator) if char_separator else row[0]
                    form2 = row[1].split(char_separator) if char_separator else row[1]
                    wordlist.append([form1, header[0]])
                    wordlist.append([form2, header[1]])
                else:
                    form = row[0].split(char_separator) if char_separator else row[0]
                    wordlist.append([form] + row[1:])
    return wordlist, header
                

def build_phone2ix(wordlist):
    # Build a mapping from symbols to integer indices
    unique_phones = sorted(set(symbol for word in wordlist for symbol in word))
    phone2ix = {phone: idx for idx, phone in enumerate(unique_phones)}
    return phone2ix

def wordlist2vec(wordlist, phone2ix, add_eos=ADD_EOS):
    forms = [torch.LongTensor(list(map(phone2ix.get, word))).to(DEVICE) for word in wordlist]
    if not add_eos:
        return forms
    else:
        eos = torch.LongTensor([0]).to(DEVICE)
        delimited = [torch.cat([form + 1, eos]) for form in forms]
        return delimited

# This is the only function that get calls from the outside.
def process_data(filepath, phone2ix=None, col_separator="\t", char_separator=" ", paired=False):
    data, header = load_file(filepath, col_separator, char_separator, header=True, paired=paired)
    wordlist, *extra = zip(*data)
    if phone2ix is None:
        phone2ix = build_phone2ix(wordlist)
    word_vec = wordlist2vec(wordlist, phone2ix, add_eos=ADD_EOS)
    return phone2ix, word_vec, extra

# end def
if __name__ == "__main__":
    file_path  = 'data/mlregtest/04.04.SL.2.1.0_TestLR.txt'
    process_data(file_path, col_separator="\t", char_separator="")

    # dataset is too big---define a function that import a large dataset as zip.
    # option: use ssh fuse
