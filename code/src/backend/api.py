from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Model & Tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./models/trained_model")
tokenizer = AutoTokenizer.from_pretrained("./models/trained_model")

# Label Mapping
label_mapping = ["Mutual Funds", "Stocks", "Crypto", "Real Estate", "Fixed Deposits"]

# Initialize FastAPI
app = FastAPI()

class QueryInput(BaseModel):
    query: str


@app.post("/query")
def get_response(input_data: QueryInput):
    inputs = tokenizer(input_data.query, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        output = model(**inputs)  # Ensure the model gets `input_ids`

    predicted_class = torch.argmax(output.logits, dim=1).item()
    recommended_product = label_mapping[predicted_class]

    return {"recommended_product": recommended_product}

# Run with:
# uvicorn backend.api:app --reload
