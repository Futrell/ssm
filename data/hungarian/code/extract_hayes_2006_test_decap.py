import pandas as pd

path = "data/hungarian/Hayes2009Wug.txt"
# Read the data from 'wug.txt'
data = pd.read_csv(path, sep="\t")

# Define a function to decapitalize a string and print if changed
def decapitalize_and_print(s):
    decapitalized = s[0].lower() + s[1:] if s and s[0].isupper() else s
    if decapitalized != s:
        print(f"Decapitalized: {s} -> {decapitalized}")
    return decapitalized

# Apply the function to the specified columns
for col in ['stem', '1st_option', '2nd_option']:
    data[col] = data[col].apply(decapitalize_and_print)

# If you want to save changes back to the file
data.to_csv(path, sep="\t", index=False)
