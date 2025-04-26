import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def init():
    global model
    global tokenizer
    global device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.getenv("AZUREML_MODEL_DIR", ".")
    # Load model and tokenizer from the model directory
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

def run(raw_data):
    try:
        # Parse input JSON
        data = json.loads(raw_data)
        # Expect input as {"inputs": ["resume text 1", "resume text 2", ...]} or a single string
        texts = data.get("inputs", data)
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize inputs
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().tolist()
            probs = probs.cpu().tolist()

        # Return predictions and probabilities
        return {
            "predictions": preds,
            "probabilities": probs
        }

    except Exception as e:
        return {"error": str(e)}
