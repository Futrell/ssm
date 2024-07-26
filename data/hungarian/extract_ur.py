import pandas as pd

def extract_allomorph(path):

	"""
	Extract allomorphs based on the given path to the CSV file.

	Args:
	- path (str): path to the .csv file.

	Returns:
	- morpheme_dict (dict): Dictionary of morphemes and corresponding allomorph counts.
	"""

	# Read the data from the .csv file
	data = pd.read_csv(path, delimiter="\t")

	# # Create a mask to find rows that need to be kept
	# mask = ~((data["Analysis"].str.contains("-PL")) & (~data["Segmentation"].str.endswith('k')))

	# # Filter the data using the mask
	# filtered_data = data[mask]

	# # Write the filtered data back to the file
	# filtered_data.to_csv(path, sep='\t', index=False)
	morpheme_dict = {}
	# breakpoint()
	# Iterate through each row
	for _, row in data.iterrows():
		morphemes = row["Analysis"].split('-')[1:]  # Skip the first morpheme
		allomorphs = row["Segmentation"].split('-')[1:]  # Skip the first allomorph

		# Only consider rows with equal splits
		if len(morphemes) == len(allomorphs):
			for morpheme, allomorph in zip(morphemes, allomorphs):
				# If morpheme already exists
				if morpheme in morpheme_dict:
					# Increase count if allomorph already exists
					if allomorph in morpheme_dict[morpheme]:
						morpheme_dict[morpheme][allomorph] += 1
					# Add new allomorph with count 1
					else:
						morpheme_dict[morpheme][allomorph] = 1
				# If morpheme doesn't exist yet, add it with the allomorph and set count to 1
				else:
					morpheme_dict[morpheme] = {allomorph: 1}

	return morpheme_dict

if __name__ == "__main__":
	# Path to the file
	path = "data/hungarian/hungarian_unimorph_lemma_freq.txt"

	# Extract allomorphs and print them
	allomorph_data = extract_allomorph(path)
	# print(allomorph_data)
	with open("data/hungarian/hungarian_allomorphs_unimorph.txt", 'w') as outfile:
		for key, inner_dict in allomorph_data.items():
			line = f"{key}: {', '.join([f'{allomorph} ({count})' for allomorph, count in inner_dict.items()])}\n"
			outfile.write(line)

	# Step 1: Handle initial j- segments
	for morpheme, allomorph_counts in allomorph_data.items():
		allomorphs = list(allomorph_counts.keys())
		if not all(allomorph.startswith('j') for allomorph in allomorphs):
			allomorph_data[morpheme] = {allomorph: count for allomorph, count in allomorph_counts.items() if not allomorph.startswith('j')}

	# Step 2: Remove shorter allomorphs
	# Adjusting the step for removing shorter allomorphs
	for morpheme, allomorph_counts in allomorph_data.items():
		max_length = max(len(allomorph) for allomorph in allomorph_counts.keys())
		allomorph_data[morpheme] = {allomorph: count for allomorph, count in allomorph_counts.items() if len(allomorph) == max_length}

	ur_dict = {}
	for morpheme, allomorph_counts in allomorph_data.items():
		
		# Step 3.1: Identify the longest allomorph length for this morpheme
		max_length = max(len(allomorph) for allomorph in allomorph_counts.keys())

		# Step 3.2: Initialize the UR for this morpheme as an empty string
		ur = ""
		
		# Step 3.3: Loop through each position
		for i in range(max_length):
			
			# Step 3.4: Collect segments at this position across all allomorphs
			segments_at_i = {allomorph[i] if i < len(allomorph) else '' for allomorph in allomorph_counts.keys()}
			
			# Step 3.5: Check segments against your conditions
			if len(segments_at_i) == 1:
				# All segments are identical
				ur += next(iter(segments_at_i))
			else:
				if segments_at_i == {"e", "o", "ö"} or segments_at_i == {"ö", "o"}:
					ur += "O"
				elif segments_at_i == {"a", "e"}:
					ur += "A"
				elif segments_at_i == {"ü", "u"}:
					ur += "U"
				elif segments_at_i == {"é", "á"}:
					ur += "Á"
				elif segments_at_i == {"ó", "ő"}:
					ur += "Ó"
				
				else:
					ur += "".join(segments_at_i)  # Assuming this is the behavior you want if none of the conditions match
			
		ur_dict[morpheme] = ur

	# Prefixing morphemes and UR with '-'
	ur_dict = {f"-{morpheme}": f"-{ur}" for morpheme, ur in ur_dict.items()}

	# print(ur_dict)

	# with open("data/hungarian/hungarian_ur_unimorph.txt", 'w') as outfile:
	# 	for morpheme, ur in ur_dict.items():
	# 		line = f"{morpheme}: {ur}\n"
	# 		outfile.write(line)




