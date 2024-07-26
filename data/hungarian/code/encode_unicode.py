input_filename = 'data/hungarian/CHILDES_merged_with_unimorph.txt'  # Replace with the path to your data file.
output_filename = 'data/hungarian/CHILDES_merged_with_unimorph.txt'  # This will be your output file with data in UTF-8 encoding.

try:
	with open(input_filename, 'rb') as f:
		data = f.read().decode('utf-8')
except UnicodeDecodeError:
	print("The file is not in UTF-8 encoding or contains characters that can't be decoded by UTF-8.")
	# You can implement further error handling here if needed.
else:
	with open(output_filename, 'w', encoding='utf-8') as f:
		f.write(data)
	print(f"Data has been written to {output_filename} in UTF-8 encoding.")
