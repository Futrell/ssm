import pandas as pd

# Load the data
data_path = "data/hungarian/Hayes2009Wug.txt"
df = pd.read_csv(data_path, sep="\t")

# Define the helper functions from the shared code
sibilant = ["s", "z", "ʃ", "ʒ", "cs", "dz", "dzs"]
coronal = ["ny", "m", "n", "l"]
bilabial = ["p", "b", "m"]
vowels = ["a", "e", "i", "o", "u", "á", "é", "í", "ó", "ö", "ő", "ú", "ü", "ű"]
unnatural = [(sibilant, "s"), (bilabial, "b"), (coronal, "n")]

def has_two_final_consonants(stem):
    multi_char_cons = ["gy", "ty", "ny", "cs", "dz", "dzs"]
    for cons in multi_char_cons:
        if stem.endswith(cons) and stem[-len(cons)-1:-len(cons)] not in vowels:
            return True
    return stem[-1] not in vowels and stem[-2] not in vowels

def match_unnatural_ending(stem):
    for char_class, return_val in unnatural:
        for char in char_class:
            if stem.endswith(char):
                return return_val
    return None

# Filter the stems
filtered_data = df[df['stem'].apply(lambda x: not (has_two_final_consonants(x) or match_unnatural_ending(x)))]

# Write to the new file
filtered_data.to_csv("data/hungarian/Hayes2009WugNatural.txt", sep="\t", index=False)
