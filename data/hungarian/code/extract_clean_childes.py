import pandas as pd

# Read the first file
df1 = pd.read_csv("data/hungarian/CHILDES_w_unimorph.txt", sep="\t")

# Read the second file
df2 = pd.read_csv("data/hungarian/CHILDES_ADULT_production.csv", usecols=["Word"])

# Filter rows in df1 whose 'SF' value is not in df2's 'Word' column
filtered_df = df1[df1['SF'].isin(df2['Word'])]

# Save the filtered data to a new file with all columns
filtered_df.to_csv("data/hungarian/CHILDES_w_unimorph_ADULT.txt", sep="\t", index=False)

print("File saved with all columns!")
