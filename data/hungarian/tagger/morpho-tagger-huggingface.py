# from transformers import AutoTokenizer

# # Load a pre-trained tokenizer; here I am using a generic multilingual model as an example.
# # You should replace "bert-base-multilingual-cased" with a Hungarian-specific model if available.
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# # Function to tokenize and segment the text
# def segment_hungarian_text(text):
#     # Tokenize the text
#     tokens = tokenizer.tokenize(text)

#     # Tokens are segmented with BERT's WordPiece, but they can be further processed if needed
#     # Depending on your requirement you can customize the processing here

#     return tokens

# # Example usage
# hungarian_text = "szeretelek"
# segments = segment_hungarian_text(hungarian_text)
# print(f"Segments for '{hungarian_text}': {segments}")

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("NYTK/morphological-generator-emmorph-mt5-hungarian")
# model = AutoModelForSeq2SeqLM.from_pretrained("NYTK/morphological-generator-emmorph-mt5-hungarian")
# from transformers import pipeline

# text2text_generator = pipeline(task="text2text-generation", model="NYTK/morphological-generator-emmorph-mt5-hungarian")

# print(text2text_generator("morph: munka [/N][Acc]")[0]["generated_text"])

# # Define the word you want to segment
# word_to_segment = "szeretelek"

# # Encode the input text
# input_ids = tokenizer.encode(f"Segment the Hungarian word: {word_to_segment}", return_tensors="pt")

# # Generate the output
# outputs = model.generate(input_ids)

# # Decode the generated ids to a text string
# segmented_word = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Print the segmented word
# print(f"Segmented word: {segmented_word}")
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# Define the parameters and hyperparameters
num_samples = 10000  # Number of samples to train on

# Initialize a tokenizer with BPE
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# Create a trainer for the tokenizer with special tokens
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])

# Path to the data txt file on disk.
data_path = 'data/hungarian/hungarian_unimorph.txt'

# Initialize lists
segmented_texts = []

# Read the dataset and prepare text for the tokenizer
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

# First, extract the texts and their segmentations and train the tokenizer
for line in lines[:min(num_samples, len(lines) - 1)]:
    parts = line.split('\t')
    if len(parts) < 5:
        continue  # Skip lines that don't have enough data
    # Use the segmented column to train the tokenizer
    segmented_text = parts[3].replace("-", " ")  # Replace hyphens with spaces
    segmented_texts.append(segmented_text)

# Train the tokenizer on the segmented texts
tokenizer.train_from_iterator(segmented_texts, trainer)

# Save the tokenizer
tokenizer.save("hungarian-bpe-tokenizer.json")

# Now you can use the tokenizer to encode unsegmented texts
test_sentence = "dobodként"
# We don't split the test sentence into subwords, the tokenizer will predict the subword units
encoded_output = tokenizer.encode(test_sentence)
print(encoded_output.tokens)
