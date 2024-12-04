import pandas as pd
import chardet


path = "data/hungarian/HungarianFromGoogleRevised.txt"
# Detecting the encoding of the file
with open(path, 'rb') as f:
    result = chardet.detect(f.read())  # or read a smaller amount f.read(10000)
    encoding = result['encoding']
print(encoding)
# Reading the data using the detected encoding
data = pd.read_csv(path, delimiter="\t", encoding=encoding)

print(data.head())  # print the first few rows to verify

# Specify the path where you want to save the output
output_path = "data/hungarian/Cleaned_HungarianData.txt"

# Write the data to a .txt file using the tab delimiter
data.to_csv(output_path, sep="\t", index=False, encoding='utf-8')
