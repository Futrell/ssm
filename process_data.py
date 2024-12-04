import numpy as np
import csv

def load(file_path, col_separator, char_separator):
    wordlist = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=col_separator)
        for row in reader:
            if row:  # Ensure the row is not empty
                # If char_separator is None, split word character by character
                if not char_separator:
                    wordlist.append(list(row[0]))
                else:
                    # Split the first column (word) using the char_separator
                    wordlist.append(row[0].split(char_separator))

    return wordlist

def build_phone2ix(wordlist):
    # Build a mapping from symbols to integer indices
    unique_phones = sorted(set(symbol for word in wordlist for symbol in word))
    phone2ix = {phone: idx + 1 for idx, phone in enumerate(unique_phones)}
    return phone2ix

def wordlist_to_vec(wordlist, phone2ix):
    return [list(map(phone2ix.get, word)) for word in wordlist]

def process_data(file_path, col_separator=",", char_separator=" "):
    wordlist = load(file_path, col_separator, char_separator)
    phone2ix = build_phone2ix(wordlist)
    word_vec = wordlist_to_vec(wordlist, phone2ix)
    return wordlist, phone2ix, word_vec
# Example usage:

if __name__ == "__main__":
    file_path  = 'space_sep.txt'
    process_data(file_path, col_separator=",", char_separator=" ")
