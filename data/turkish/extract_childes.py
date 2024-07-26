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
	directory_path = 'data/turkish/CHILDES'
	output_path = 'data/turkish/CHILDES_CHI_production.csv'

	df = merge_words_from_directory(directory_path, participant='CHI')
	df.to_csv(output_path, index=False)



