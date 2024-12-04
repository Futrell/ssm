import pandas as pd

# Define Hungarian alphabet
hungarian_alphabet = "aábcdeéfghiíjklmnoóöőpqrstuúüűvxyz"

# Read the datasets
childes_df = pd.read_csv('data/hungarian/CHILDES_ADULT_production.csv')
unimorph_df = pd.read_csv('data/hungarian/hungarian_unimorph.txt', delimiter='\t')

# Decapitalize, remove numbers, non-Hungarian characters, and words with less than 4 segments
childes_df['Word'] = childes_df['Word'].str.lower()
childes_df['Word'] = childes_df['Word'].apply(lambda x: ''.join([char for char in x if char in hungarian_alphabet]))
childes_df = childes_df[childes_df['Word'].str.len() >= 4]

# Extract stems and their labels
stem_dict = dict(zip(unimorph_df['Segmentation'].str.split('-').str[0], unimorph_df['Analysis'].str.split('-').str[0]))
stems = sorted(stem_dict.keys(), key=len, reverse=True)

# Extract suffix morphemes and their labels
morphemes = unimorph_df['Segmentation'].str.split('-', expand=True).iloc[:, 1:]
labels = unimorph_df['Analysis'].str.split('-', expand=True).iloc[:, 1:]

suffixes = sorted(set(morpheme for sublist in morphemes.values for morpheme in sublist if pd.notna(morpheme)), key=len, reverse=True)

suffix_dict = {}
for idx, row in morphemes.iterrows():
	for col, morpheme in enumerate(row):
		if pd.notna(morpheme):
			label = labels.iat[idx, col]
			suffix_dict[morpheme] = label

def extract_suffixes(segment, initial_stem, suffixes, suffix_dict):
	suffix_list, label_list = [], []
	
	while segment:
		matched_suffix, matched_label = None, None
		for suffix in suffixes:
			if segment.endswith(suffix):
				matched_suffix = suffix
				matched_label = suffix_dict.get(suffix, suffix)
				break
		
		if matched_suffix:
			suffix_list.insert(0, matched_suffix)
			label_list.insert(0, matched_label)
			segment = segment[:-len(matched_suffix)]
		else:
			# If the suffix isn't recognized, assume it's part of the stem
			initial_stem += segment[0]
			segment = segment[1:]

	return suffix_list, label_list, initial_stem  # Return the updated stem as well


results = []
unlabeled_suffixes = []

for _, row in childes_df.iterrows():
	word = row['Word']
	freq = row['Frequency']

	# Checkpoint 1: Is the word "meggyógyitom"?
	if word == "meggyógyitom":
		print("Checkpoint 1: Processing word:", word)

	for stem in stems:

		if word.startswith(stem):
			remaining = word[len(stem):]
			
			# Checkpoint 2: Found stem for the word "meggyógyitom"
			if word == "meggyógyitom":
				print("Checkpoint 2: Identified stem:", stem)
				print("Remaining segment:", remaining)
				
			suffix_list, label_list, updated_stem = extract_suffixes(remaining, stem, suffixes, suffix_dict)  # Capture the updated stem


			# Checkpoint 3: Extracted suffixes for the word "meggyógyitom"
			if word == "meggyógyitom":
				print("Checkpoint 3: Extracted Suffixes:", suffix_list)
				print("Extracted Labels:", label_list)

			if suffix_list:
				results.append({
							'SF': word,
							'Freq': freq,
							'Lemma': updated_stem,  # Use the updated stem
							'Analysis': stem_dict.get(updated_stem, updated_stem) + '-' + '-'.join(label_list),  # Use the updated stem
							'Segmentation': updated_stem + '-' + '-'.join(suffix_list)  # Use the updated stem
				})                


				for suffix, label in zip(suffix_list, label_list):
					if label == suffix:
						unlabeled_suffixes.append({'Word': word, 'Suffix': suffix})
			break
stem_suffix_df = pd.DataFrame(results)

# Print the words where suffixes do not have a corresponding label
unlabeled_suffixes_df = pd.DataFrame(unlabeled_suffixes)
print("Unlabeled Suffixes:")
print(unlabeled_suffixes_df)

# Save to a csv file
stem_suffix_df.to_csv('data/hungarian/CHILDES_ADULT_production_notin_unimorph.csv', index=False)

# Read the datasets again for merging
unimorph_chi_df = pd.read_csv('data/hungarian/CHILDES_w_unimorph_ADULT_cleaned.txt', delimiter='\t')
childes_notin_unimorph_df = pd.read_csv('data/hungarian/CHILDES_ADULT_production_notin_unimorph.csv', delimiter=',')

# Merge the two dataframes
merged_df = pd.concat([unimorph_chi_df, childes_notin_unimorph_df], ignore_index=True)

# Drop duplicate rows, keeping the first occurrence
merged_df = merged_df.drop_duplicates(subset=['SF'])

# Decapitalize certain columns
columns_to_replace = ['Segmentation', 'SF', 'Lemma']
for col in columns_to_replace:
    merged_df[col] = merged_df[col].str.lower()

# Incorporating the modify_analysis function
def modify_analysis(row):
    morphemes = row["Analysis"].split('-')[1:]  # Skip the first morpheme
    allomorphs = row["Segmentation"].split('-')[1:]  # Skip the first allomorph

    new_morphemes = []
    
    for morpheme, allomorph in zip(morphemes, allomorphs):
        if morpheme == "IND.PRS.INDF.2.SG":
            if allomorph.endswith('l'):
                morpheme += ".Ol"
            elif allomorph.endswith('z'):
                morpheme += ".asz"
        new_morphemes.append(morpheme)

    # Reconstruct the modified analysis string
    return '-'.join([row["Analysis"].split('-')[0]] + new_morphemes)

# Apply the modify_analysis function to the merged_df dataframe
merged_df["Analysis"] = merged_df.apply(modify_analysis, axis=1)

# Save the modified dataframe back to the same path
merged_df.to_csv('data/hungarian/CHILDES_w_unimorph_ADULT_labelled_missing.txt', index=False, sep='\t')

print(merged_df)






