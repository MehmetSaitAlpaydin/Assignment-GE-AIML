import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
import pandas as pd

# Define available categories with numeric labels
categories = {
    "18828_alt.atheism": 0, "18828_comp.graphics": 1, "18828_comp.os.ms-windows.misc": 2,
    "18828_comp.sys.ibm.pc.hardware": 3, "18828_comp.sys.mac.hardware": 4, "18828_comp.windows.x": 5,
    "18828_misc.forsale": 6, "18828_rec.autos": 7, "18828_rec.motorcycles": 8, "18828_rec.sport.baseball": 9,
    "18828_rec.sport.hockey": 10, "18828_sci.crypt": 11, "18828_sci.electronics": 12, "18828_sci.med": 13,
    "18828_sci.space": 14, "18828_soc.religion.christian": 15, "18828_talk.politics.guns": 16,
    "18828_talk.politics.mideast": 17, "18828_talk.politics.misc": 18, "18828_talk.religion.misc": 19
}

# Reload dataset
datasets = []
for category, label in categories.items():
    ds = load_dataset("newsgroup", name=category, split="train")
    ds = ds.add_column("label", [label] * len(ds))  # Assign numeric label
    datasets.append(ds)

# Concatenate all datasets
dataset = concatenate_datasets(datasets)

# Convert dataset to DataFrame
df = pd.DataFrame({'text': dataset['text'], 'label': dataset['label']})

# Split dataset (same as in training)
_, val_texts, _, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

# Load trained model and tokenizer
model_path = "distilbert_newsgroup_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize validation dataset in batches
batch_size = 8  # Reduce if OOM persists
predictions, true_labels = [], []

for i in range(0, len(val_texts), batch_size):
    batch_texts = val_texts[i : i + batch_size]
    batch_labels = val_labels[i : i + batch_size]

    encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    labels = torch.tensor(batch_labels).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        logits = model(**encodings).logits
        preds = torch.argmax(logits, dim=-1)

    predictions.extend(preds.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")

# Print evaluation results
print("Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
