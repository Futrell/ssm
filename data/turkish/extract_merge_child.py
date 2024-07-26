# File paths
childes_path = 'data/turkish/childes.txt'
childes_adult_train_path = 'data/turkish/childes_adult_train.txt'
childes_child_test_path = 'data/turkish/childes_child_test.txt'
childes_new_path = 'data/turkish/childes_new.txt'

# Helper function to convert a line to key (excluding frequency)
def line_to_key(line):
    return tuple(line.strip().split('\t')[:-1])

# Read the files and skip the header
with open(childes_path, 'r') as f:
    header = f.readline()  # store the header
    childes_data = {line_to_key(line): int(line.strip().split('\t')[-1]) for line in f.readlines()}

with open(childes_adult_train_path, 'r') as f:
    _ = f.readline()  # skip the header
    for line in f.readlines():
        key = line_to_key(line)
        if key not in childes_data:
            childes_data[key] = 1

with open(childes_child_test_path, 'r') as f:
    _ = f.readline()  # skip the header
    for line in f.readlines():
        key = line_to_key(line)
        if key not in childes_data:
            childes_data[key] = 1

# Convert back to line format and sort by frequency
merged_data_sorted = sorted([('\t'.join(key) + '\t' + str(freq) + '\n') for key, freq in childes_data.items()],
                            key=lambda x: int(x.strip().split('\t')[-1]), reverse=True)

# Write merged data to a new file
with open(childes_new_path, 'w') as f:
    f.write(header)
    f.writelines(merged_data_sorted)
