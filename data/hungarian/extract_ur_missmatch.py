import pandas as pd

# Read the first file
df1 = pd.read_csv('data/hungarian/hungarian_allomorphs_unimorph.txt', sep='\t', header=None, names=['Key', 'Value'])
df1 = df1.drop_duplicates(subset='Key')

# Read the second file
df2 = pd.read_csv('data/hungarian/hungarian_ur_unimorph.txt', sep='\t', header=None, names=['Key', 'Value'])
df2['Key'] = df2['Key'].str.lstrip('-')

# Merge dictionaries
dict1 = dict(zip(df1['Key'], df1['Value']))
for key, value in dict1.items():
    if key not in df2['Key'].values:
        # Append new entry with leading '-' in the key
        df2 = df2.append({'Key': f"-{key}", 'Value': value}, ignore_index=True)

# Add '-' back to the keys in df2
df2['Key'] = '-' + df2['Key']

# Sort and save the updated second file
output_path = 'data/hungarian/hungarian_ur_unimorph_updated.txt'
df2.sort_values(by='Key').to_csv(output_path, sep='\t', index=False, header=False)

