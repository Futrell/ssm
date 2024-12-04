import random

def _split_data(data, split, seed=100):  # default seed value set to 100
    random.seed(seed)  # set the seed for reproducibility
    sorted_data = sorted(data, key=lambda x: int(x['Freq']), reverse=True)

    # Shuffle the data using the set seed
    random.shuffle(sorted_data)

    split_idx = int(split * len(sorted_data))
    train_data = sorted_data[:split_idx]
    dev_data = sorted_data[split_idx:]
    return train_data, dev_data

def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as file:
        next(file)  # Skip header
        for line in file:
            SF, Freq, Lemma, Analysis, Segmentation = line.strip().split('\t')
            data_item = {
                'SF': SF,
                'Freq': Freq,
                'Lemma':Lemma,
                'Analysis': Analysis,
                'Segmentation': Segmentation,
            }
            data.append(data_item)
    return data

def write_data_to_file(filename, data):
    with open(filename, 'w') as file:
        # Write header
        file.write("SF\tFreq\tLemma\tSegmentation\tAnalysis\n")
        for item in data:
            line = f"{item['SF']}\t{item['Freq']}\t{item['Lemma']}\t{item['Segmentation']}\t{item['Analysis']}\n"
            file.write(line)

if __name__ == "__main__":
    data = read_data_from_file('data/hungarian/CHILDES_w_unimorph_ADULT_labelled_missing.txt')
    train_data, test_data = _split_data(data, 0.8)  # Splitting 80% for training and 20% for testing
    write_data_to_file('data/hungarian/CHILDES_w_unimorph_ADULT_test_tagged.txt', test_data)
    write_data_to_file('data/hungarian/CHILDES_w_unimorph_ADULT_train_tagged.txt', train_data)
