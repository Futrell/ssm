import pandas as pd

# Reading the data from both morpho files and combine them
morpho1 = pd.read_csv('data/turkish/morpho.txt', sep='\t')
morpho2 = pd.read_csv('data/turkish/childes.txt', sep='\t')
morpho = pd.concat([morpho1, morpho2]).drop_duplicates()

# Reading the data from adult and child production files
adult = pd.read_csv('data/turkish/CHILDES_ADULT_production.csv')
child = pd.read_csv('data/turkish/CHILDES_CHI_production.csv')

# Find intersection for adult and morpho
intersection_adult = morpho[morpho['Word'].isin(adult['Word'])].copy()
# Replace frequency in intersection_adult
intersection_adult['Freq'] = intersection_adult['Word'].map(adult.set_index('Word')['Frequency'])

# Find intersection for child and morpho
intersection_child = morpho[morpho['Word'].isin(child['Word'])].copy()
# Replace frequency in intersection_child
intersection_child['Freq'] = intersection_child['Word'].map(child.set_index('Word')['Frequency'])

# Find rows in intersection_child that are not present in intersection_adult
unique_child = intersection_child[~intersection_child['Word'].isin(intersection_adult['Word'])].copy()

# Remove duplicates and sort dataframes by frequency
intersection_adult = intersection_adult.drop_duplicates().sort_values(by='Freq', ascending=False)
intersection_child = intersection_child.drop_duplicates().sort_values(by='Freq', ascending=False)
unique_child = unique_child.drop_duplicates().sort_values(by='Freq', ascending=False)

# Save the updated datasets
intersection_adult.to_csv('data/turkish/childes_adult_train.txt', sep='\t', index=False)
intersection_child.to_csv('data/turkish/childes_child_test.txt', sep='\t', index=False)
unique_child.to_csv('data/turkish/childes_child_test_noadult.txt', sep='\t', index=False)

# Words in adult and child not in morpho
missing_adult = adult[~adult['Word'].isin(morpho['Word'])]
missing_child = child[~child['Word'].isin(morpho['Word'])]

# Save these to separate files
missing_adult.to_csv('data/turkish/missing_from_morpho_adult.txt', sep='\t', index=False)
missing_child.to_csv('data/turkish/missing_from_morpho_child.txt', sep='\t', index=False)


# this code extract the intersection between the raw CHILDES data and MorphoChallenge and Belth (2023) data to obtain the morphological analysis