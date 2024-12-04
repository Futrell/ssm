import pandas as pd

# Define the path to the file
path = "data/hungarian/hun-master/hun.segmentations"

# Read the data from the .txt file
data = pd.read_csv(path, delimiter="\t", header=None, names=["Lemma", "SF", "Analysis", "Segmentation"])

# Filter out rows with "#Hungarian-entryway" early (assuming it mostly appears in the 'Segmentation' column)
data = data[~data['Lemma'].str.contains("#")]

# Replace | with - and ; with . in the 'Analysis' and 'Segmentation' columns
data["Analysis"] = data["Analysis"].str.replace(";", ".").str.replace("|", "-")
data["Segmentation"] = data["Segmentation"].str.replace("|", "-")

# For rows where 'Segmentation' is just "-", replace with corresponding SF value
mask = data["Segmentation"] == "-"
data.loc[mask, "Segmentation"] = data.loc[mask, "SF"]

# ... [Rest of your code]
# Check where 'Segmentation' and 'Analysis' have a different number of '-'
mask_diff = data["Segmentation"].str.count("-") != data["Analysis"].str.count("-")

# Count the number of such rows
count_removed = mask_diff.sum()

# Remove the rows
data = data[~mask_diff]

# Report the number of rows removed
print(f"Removed {count_removed} rows with mismatched '-' counts.")

# Add a new column for SF frequency
data['SF_Frequency'] = 1

# Write the transformed data to a new .txt file
output_path = "data/hungarian/hungarian_unimorph.txt"
data.to_csv(output_path, sep="\t", index=False)
