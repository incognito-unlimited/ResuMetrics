import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Global variables for model, tokenizer, and label mapping
model = None
tokenizer = None
label_mapping = None
device = None

def init():
    global model, tokenizer, label_mapping, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load label encoder classes
    label_file = os.path.join(model_dir, "label_encoder_classes.txt")
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    label_mapping = {i: label for i, label in enumerate(labels)}

def run(raw_data):
    try:
        data = json.loads(raw_data)
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

        # Map predicted indices to label names
        pred_labels = [label_mapping.get(p, str(p)) for p in preds]

        return {
            "predictions": pred_labels,
            "probabilities": probs
        }

    except Exception as e:
        return {"error": str(e)}
