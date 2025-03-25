import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv("data/dataset.csv")

# Create a text feature from relevant columns
df["text"] = df.apply(lambda row: f"{row['age']} {row['annual_income']} {row['credit_score']} {row['investment_portfolio']}", axis=1)

# Encode Labels
label_encoder = LabelEncoder()
df["preferred_investment_type"] = label_encoder.fit_transform(df["preferred_investment_type"])

# Train-Test Split (80% Training, 20% Testing)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["preferred_investment_type"].tolist(), test_size=0.2, random_state=42
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize Data
# Tokenize Data with explicit return format
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt")


# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "labels": train_labels})
val_dataset = Dataset.from_dict({"input_ids": val_encodings["input_ids"], "labels": val_labels})

# Load Pretrained Model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(set(train_labels)))

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./models/trained_model",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    logging_dir='./logs',
    load_best_model_at_end=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train Model
trainer.train()

# Save Model
model.save_pretrained("./models/trained_model")
tokenizer.save_pretrained("./models/trained_model")

print("✅ Training complete! Model saved.")
