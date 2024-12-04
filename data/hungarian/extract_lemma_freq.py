import pandas as pd

# Assuming the file is structured as provided in the user input
file_path = 'data/hungarian/hungarian_unimorph.txt'

df = pd.read_csv(file_path, sep="\t")

# Drop duplicate rows based on 'SF' column
df = df.drop_duplicates(subset=['SF'])

# Add a new column 'Freq' for lemma frequency
df['Freq'] = df.groupby('Lemma')['Lemma'].transform('count')

# Sort the DataFrame based on 'Freq' in descending order
df_sorted = df.sort_values(by='Freq', ascending=False)

# Output the sorted DataFrame to a new file
output_file = 'data/hungarian/hungarian_unimorph_lemma_freq.txt'
df_sorted.to_csv(output_file, sep="\t", index=False)
