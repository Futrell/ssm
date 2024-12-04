from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForTokenClassification, BertConfig, Trainer, TrainingArguments

# Define paths
data_path = 'data/hungarian/hungarian_unimorph.txt'
output_tokenizer_path = 'hungarian-bpe-tokenizer.json'

# Load and preprocess data
with open(data_path, 'r', encoding='utf-8') as f:
    lines = [line.strip().split('\t') for line in f if line.strip()]

# Split into segments and tags
data = [(parts[3].replace("-", " "), parts[2]) for parts in lines if len(parts) >= 5]
segments, tags = zip(*data)

# Initialize a tokenizer with BPE
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# Create a trainer for the tokenizer with special tokens
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
tokenizer.train_from_iterator(segments, trainer)

# Save the tokenizer
tokenizer.save(output_tokenizer_path)

# Load tokenizer with Hugging Face bindings
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=output_tokenizer_path)

# Define a PyTorch dataset
class HungarianDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=512)
        item['labels'] = self.labels[idx]
        return item

# Split the dataset into training and validation sets
train_texts, val_texts, train_tags, val_tags = train_test_split(segments, tags, test_size=0.1)

# Define tag to index mapping and reverse mapping
unique_tags = set(tag for sublist in tags for tag in sublist.split())
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

# Convert tags to IDs
train_tags = [[tag2id[tag] for tag in doc.split()] for doc in train_tags]
val_tags = [[tag2id[tag] for tag in doc.split()] for doc in val_tags]

# Create datasets
train_dataset = HungarianDataset(train_texts, train_tags, hf_tokenizer)
val_dataset = HungarianDataset(val_texts, val_tags, hf_tokenizer)

# Load a pre-trained BERT model for token classification
config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_labels=len(unique_tags))
model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', config=config)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and evaluate
trainer.train()
trainer.evaluate()

# Predict on new data
test_sentence = "temetöt"
test_encoding = hf_tokenizer(test_sentence, return_tensors='pt')

# Predict tags
with torch.no_grad():
    logits = model(**test_encoding).logits

predictions = torch.argmax(logits, dim=-1)
predicted_tags = [id2tag[id.item()] for id in predictions[0]]

print("Predicted tags:", predicted_tags)
