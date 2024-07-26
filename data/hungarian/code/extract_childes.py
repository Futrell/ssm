import os
import pylangacq as pla
import pandas as pd
import string

def extract_words_from_cha(directory_path, participant):
	chat = pla.read_chat(directory_path)
	if participant == 'ALL':
		return chat.words(by_utterances=True)
	if participant == 'ADULT':
		return chat.words(exclude="CHI", by_utterances=True)
	if participant == 'CHI':
		return chat.words(participants="CHI", by_utterances=True)
	return []

def merge_words_from_directory(root_dir,participant):
	all_words = []

	# Recursively walk through the directory and get all .cha files
	for dirpath, dirnames, filenames in os.walk(root_dir):
		for file in filenames:
			if file.endswith('.cha'):
				file_path = os.path.join(dirpath, file)
				words = extract_words_from_cha(file_path,participant)
				all_words.extend(words)

	# Convert the words into a single string with newlines separating them
	merged_words = "\n".join([" ".join(utterance) for utterance in all_words])

	# Convert the string to a DataFrame
	df = pd.DataFrame({'Words': merged_words.split("\n")})

	return df



def merge_words_from_directory(root_dir, participant):
	all_words = []

	# Recursively walk through the directory and get all .cha files
	for dirpath, dirnames, filenames in os.walk(root_dir):
		for file in filenames:
			if file.endswith('.cha'):
				file_path = os.path.join(dirpath, file)
				words = extract_words_from_cha(file_path, participant)
				all_words.extend(words)

	# Convert the words into a single string with newlines separating them
	merged_words = "\n".join([" ".join(utterance) for utterance in all_words])

	# Convert the string to a DataFrame
	df = pd.DataFrame({'Words': merged_words.split("\n")})

	# Split each line into words and remove special punctuation
	words_list = []
	for index, row in df.iterrows():
		words = row['Words'].split()
		for word in words:
			clean_word = word.translate(str.maketrans('', '', string.punctuation))
			if clean_word:  # Ensure the word isn't empty after cleaning
				words_list.append(clean_word)

	# Convert words_list into a DataFrame
	word_df = pd.DataFrame(words_list, columns=['Word'])

	# Count the token frequency for each word
	word_df['Frequency'] = word_df.groupby('Word')['Word'].transform('count')

	# Drop duplicate words
	word_df = word_df.drop_duplicates().reset_index(drop=True)

	return word_df



if __name__ == "__main__":
	# directory_path = 'data/hungarian/CHILDES'
	# output_path = 'data/hungarian/CHILDES_ADULT_production.csv'

	# df = merge_words_from_directory(directory_path, participant='ADULT')
	# df.to_csv(output_path, index=False)


	# # Read the word_df (assuming you've saved it from the previous steps)
	# word_df = pd.read_csv(output_path)

	# # Read the hungarian_unimorph.txt dataset
	# unimorph_df = pd.read_csv('data/hungarian/hungarian_unimorph.txt', delimiter='\t')

	# # Filter word_df using the SF in unimorph_df
	# word_df = word_df[word_df['Word'].isin(unimorph_df['SF'])]

	# # Merge Lemma, SF, Analysis, and Segmentation columns based on 'Word' and 'SF'
	# merged_df = pd.merge(word_df, unimorph_df[['Lemma', 'SF', 'Analysis', 'Segmentation']], 
	# 					left_on='Word', right_on='SF', how='left')

	# # Drop the SF column (since it's the same as 'Word' column)
	# merged_df.drop('SF', axis=1, inplace=True)

	# # Save the merged dataframe
	# merged_df.to_csv(f"{output_path}unimorph.csv", index=False)


	# Read the CSV
	path = "data/hungarian/hungarian_unimorph_lemma_freq.txt"
	data = pd.read_csv(path, delimiter="\t")

	# Define a function to modify the Analysis column
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
			# if morpheme == "IND.PRS.DEF.3.PL":
			# 	if allomorph.endswith('ik'):
			# 		morpheme += ".ik"
			# if morpheme == "IND.PRS.DEF.3.SG":
			# 	if allomorph.endswith('i'):
			# 		morpheme += "(-i)"
			new_morphemes.append(morpheme)

		# Reconstruct the modified analysis string
		return '-'.join([row["Analysis"].split('-')[0]] + new_morphemes)

	# Apply the modify_analysis function
	data["Analysis"] = data.apply(modify_analysis, axis=1)

	# Save the modified dataframe back to CSV
	data.to_csv(path, sep=',', index=False)


