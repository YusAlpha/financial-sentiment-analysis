import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("train_data.csv")

train_data['text'] = train_data['text'].astype(str).fillna("")

train_finetune, val_finetune = train_test_split(train_data, test_size=0.2, stratify=train_data['label'], random_state=66)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=3).to(device)

class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0).to(device),
            "attention_mask": encoding["attention_mask"].squeeze(0).to(device),
            "labels": torch.tensor(label, dtype=torch.long).to(device),
        }

label_map = {"Neutral": 0, "Bullish": 1, "Bearish": 2}

train_finetune['text'] = train_finetune['text'].astype(str).fillna("")
val_finetune['text'] = val_finetune['text'].astype(str).fillna("")

train_labels = train_finetune['label'].map(label_map).values
val_labels = val_finetune['label'].map(label_map).values

print("Processing training dataset...")
train_dataset = FinancialSentimentDataset(
    list(tqdm(train_finetune['text'], desc="Tokenizing Train Data")),
    train_labels,
    finbert_tokenizer
)

print("Processing validation dataset...")
val_dataset = FinancialSentimentDataset(
    list(tqdm(val_finetune['text'], desc="Tokenizing Validation Data")),
    val_labels,
    finbert_tokenizer
)

training_args = TrainingArguments(
    output_dir="./finbert_trained",
    num_train_epochs=5,  
    per_device_train_batch_size=48, 
    gradient_accumulation_steps=1, 
    per_device_eval_batch_size=32,
    learning_rate=5e-6, 
    max_grad_norm=1.0,
    warmup_steps=50, 
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="no",
)

trainer = Trainer(
    model=finbert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, 
)

print("Starting FinBERT fine-tuning...")
trainer.train()

finbert_model.save_pretrained("./finbert_trained")
finbert_tokenizer.save_pretrained("./finbert_trained")

print("FinBERT fine-tuning complete. Model saved successfully.")
