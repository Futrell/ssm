import pandas as pd

# Define Turkish alphabet
turkish_alphabet = "abcçdefgğhıijklmnoöpqrsştuüvwxyz"

# Read the datasets
childes_df = pd.read_csv('data/turkish/CHILDES_CHI_production.csv')
morpho = pd.read_csv('data/turkish/morpho.txt', sep='\t')

# Decapitalize, remove numbers, non-Turkish characters, and words with less than 4 segments
childes_df['Word'] = childes_df['Word'].str.lower()
childes_df['Word'] = childes_df['Word'].apply(lambda x: ''.join([char for char in x if char in turkish_alphabet]))

# Extract stems and their labels from morpho
stem_dict = dict(zip(morpho['Segmentation'], morpho['Analysis']))

# We will attempt to match stems first before moving to suffixes
stems = sorted(stem_dict.keys(), key=len, reverse=True)

results = []
unlabeled_suffixes = []

for _, row in childes_df.iterrows():
    word = row['Word']
    freq = row['Frequency']

    for stem in stems:
        if word.startswith(stem):
            remaining = word[len(stem):]
            labels = stem_dict[stem]
            
            results.append({
                'SF': word,
                'Freq': freq,
                'Word': stem,
                'Segmentation': stem + '-' + remaining if remaining else stem,
                'Analysis': labels if remaining else labels.split('-')[0]
            })
            break

stem_suffix_df = pd.DataFrame(results)

# Save to a csv file
stem_suffix_df.to_csv('data/turkish/CHILDES_CHI_production_notin_unimorph.csv', index=False)

# Read the datasets again for merging
unimorph_chi_df = pd.read_csv('data/turkish/CHILDES_w_unimorph_CHI.txt', delimiter='\t')
childes_notin_morpho = pd.read_csv('data/turkish/CHILDES_CHI_production_notin_unimorph.csv', delimiter=',')

# Merge the two dataframes
merged_df = pd.concat([unimorph_chi_df, childes_notin_morpho], ignore_index=True)

# Remove duplicate rows
merged_df = merged_df.drop_duplicates()

merged_df.to_csv('data/turkish/CHILDES_w_unimorph_CHI_new.txt', index=False, sep='\t')

print(merged_df)
