import random

def _split_data(data, split, seed=100):  # default seed value set to 100
    random.seed(seed)  # set the seed for reproducibility
    sorted_data = sorted(data, key=lambda x: int(x['Freq']), reverse=True)

    # Shuffle the data using the set seed
    # random.shuffle(sorted_data)

    split_idx = int(split * len(sorted_data))
    train_data = sorted_data[:split_idx]
    dev_data = sorted_data[split_idx:]
    return train_data, dev_data

def _split_data(data, split, seed=10):
    random.seed(seed)  # set the seed for reproducibility

    # Calculate total frequency to normalize the probabilities
    total_freq = sum(int(x['Freq']) for x in data)
    probabilities = [int(x['Freq']) / total_freq for x in data]

    # Sample without replacement for the train data
    train_size = int(split * len(data))
    train_indices = random.choices(range(len(data)), weights=probabilities, k=train_size)
    train_data = [data[i] for i in train_indices]

    # Remove duplicates from train_indices to ensure dev_data does not contain duplicates
    unique_train_indices = set(train_indices)

    # The rest of the data will be for development
    dev_data = [item for index, item in enumerate(data) if index not in unique_train_indices]

    return train_data, dev_data

def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as file:
        next(file)  # Skip header
        for line in file:
            SF, Word, Segmentation, Analysis, Freq = line.strip().split('\t')
            data_item = {
                'SF': SF,
                'Word': Word,
                'Segmentation': Segmentation,
                'Analysis': Analysis,
                'Freq': Freq
            }
            data.append(data_item)
    return data

def write_data_to_file(filename, data):
    with open(filename, 'w') as file:
        # Write header
        file.write("SF\tWord\tSegmentation\tAnalysis\tFreq\n")
        for item in data:
            line = f"{item['SF']}\t{item['Word']}\t{item['Segmentation']}\t{item['Analysis']}\t{item['Freq']}\n"
            file.write(line)

if __name__ == "__main__":
    data = read_data_from_file('data/turkish/morpho.txt')
    train_data, test_data = _split_data(data, 0.14564194488)  # Splitting 80% for training and 20% for testing
    write_data_to_file('data/turkish/morpho_train.txt', train_data)
    write_data_to_file('data/turkish/morpho_test.txt', test_data)
    # data = read_data_from_file('data/turkish/childes.txt')
    # train_data, test_data = _split_data(data, 0.8)  # Splitting 80% for training and 20% for testing
    # write_data_to_file('data/turkish/childes_train.txt', train_data)
    # write_data_to_file('data/turkish/childes_test.txt', test_data)
