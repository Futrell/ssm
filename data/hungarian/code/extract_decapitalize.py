import pandas as pd

# Define the path for easier reference
data_path = "data/hungarian/CHILDES_w_unimorph_CHI_cleaned.txt"

# Read the data
df = pd.read_csv(data_path, sep='\t')  # assuming the data is tab-separated

import pandas as pd

# Define the path for easier reference
data_path = "data/hungarian/CHILDES_w_unimorph_ADULT_cleaned.txt"

# Read the data
df = pd.read_csv(data_path, sep='\t')  # assuming the data is tab-separated

# Decapitalize all characters in the specified columns
columns_to_replace = ['Segmentation', 'SF', 'Lemma']
for col in columns_to_replace:
    df[col] = df[col].str.lower()

# Output the modified dataframe back to the original file
df.to_csv(data_path, sep='\t', index=False)


# Output the modified dataframe back to the original file
df.to_csv(data_path, sep='\t', index=False)
