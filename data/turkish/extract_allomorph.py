import pandas as pd
import pprint as pp
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
	path = "data/turkish/childes.txt"

	# Extract allomorphs and print them
	allomorph_data = extract_allomorph(path)
	pp.pprint(allomorph_data)