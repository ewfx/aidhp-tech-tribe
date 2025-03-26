from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split


# Load Data
df = pd.read_csv("data/dataset.csv")
df["text"] = df.apply(lambda row: f"{row['age']} {row['annual_income']} {row['credit_score']} {row['investment_portfolio']}", axis=1)

# Encode Labels
label_encoder = LabelEncoder()
df["preferred_investment_type"] = label_encoder.fit_transform(df["preferred_investment_type"])

# Split Data (Use same 20% Test Split)
_, test_texts, _, test_labels = train_test_split(
    df["text"].tolist(), df["preferred_investment_type"].tolist(), test_size=0.2, random_state=42
)

# Load Model & Tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./models/trained_model")
tokenizer = AutoTokenizer.from_pretrained("./models/trained_model")



test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt")

# Model inference
with torch.no_grad():
    outputs = model(**test_encodings)  # Ensure input_ids are provided
predictions = torch.argmax(outputs.logits, dim=1)


# Calculate Accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Generate Classification Report
report = classification_report(test_labels, predictions, target_names=label_encoder.classes_)
print(report)
