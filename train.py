import torch
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, concatenate_datasets
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"

# Define available categories with numeric labels
categories = {
    "18828_alt.atheism": 0, "18828_comp.graphics": 1, "18828_comp.os.ms-windows.misc": 2,
    "18828_comp.sys.ibm.pc.hardware": 3, "18828_comp.sys.mac.hardware": 4, "18828_comp.windows.x": 5,
    "18828_misc.forsale": 6, "18828_rec.autos": 7, "18828_rec.motorcycles": 8, "18828_rec.sport.baseball": 9,
    "18828_rec.sport.hockey": 10, "18828_sci.crypt": 11, "18828_sci.electronics": 12, "18828_sci.med": 13,
    "18828_sci.space": 14, "18828_soc.religion.christian": 15, "18828_talk.politics.guns": 16,
    "18828_talk.politics.mideast": 17, "18828_talk.politics.misc": 18, "18828_talk.religion.misc": 19
}

# Load all categories separately and assign numeric labels
datasets = []
for category, label in categories.items():
    ds = load_dataset("newsgroup", name=category, split="train")
    ds = ds.add_column("label", [label] * len(ds))  # Assign numeric label
    datasets.append(ds)

# Concatenate all datasets
dataset = concatenate_datasets(datasets)

# Convert dataset to DataFrame
df = pd.DataFrame({'text': dataset['text'], 'label': dataset['label']})

# Split into train and test sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512)

# Tokenize datasets
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(categories))

# Move model to GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Custom logging callback
class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 10 == 0:  # Log every 10 steps
            print(f"Step {state.global_step}: {logs}")

# Training arguments for GPU
training_args = TrainingArguments(
    output_dir="./results", eval_strategy="epoch", save_strategy="epoch", per_device_train_batch_size=16,
    per_device_eval_batch_size=16, num_train_epochs=3, weight_decay=0.01, push_to_hub=False,
    report_to="none", logging_dir="./logs", logging_steps=10, disable_tqdm=False)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer setup
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
    tokenizer=tokenizer, data_collator=data_collator, callbacks=[LogCallback()]
)

# Train the model
print("Starting training...")
start_time = time.time()
trainer.train()
print(f"Training complete! Total time: {time.time() - start_time:.2f} seconds")

# Save the model
model.save_pretrained("distilbert_newsgroup_model")
tokenizer.save_pretrained("distilbert_newsgroup_model")

print("Model saved!")