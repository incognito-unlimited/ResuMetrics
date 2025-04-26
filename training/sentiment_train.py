import os
import argparse
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import nltk
from sklearn.preprocessing import LabelEncoder
import transformers.integrations

# Disable Hugging Face MLflow autologging globally
transformers.integrations.MLflowCallback._active = False

nltk.download('punkt')

def main(args):
    # Debug: Print data path
    print(f"Loading training data from: {args.data}")
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Encode labels (negative=0, positive=1, neutral=2)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['sentiment'])
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df[['feedback', 'label']])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["feedback"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["feedback"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    # Split dataset
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3  # For positive, negative, neutral
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=os.path.join(args.model_path, "logs"),
        logging_steps=10,
    )
    
    # Initialize trainer normally
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda eval_pred: {
            "accuracy": (eval_pred.predictions.argmax(-1) == eval_pred.label_ids).mean()
        }
    )
    
    # Remove MLflowCallback from trainer callbacks to avoid parameter size limit error
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if not isinstance(cb, transformers.integrations.MLflowCallback)
    ]
    
    # Train model
    trainer.train()
    
    # Determine output path for saving model artifacts (prefer AZUREML_OUTPUT_DIR)
    output_path = os.environ.get("AZUREML_OUTPUT_DIR", args.model_path)
    print(f"Saving model and tokenizer to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Save model and tokenizer explicitly to output_path
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save label encoder classes
    label_encoder_path = os.path.join(output_path, "label_encoder_classes.txt")
    with open(label_encoder_path, "w") as f:
        for label in label_encoder.classes_:
            f.write(f"{label}\n")
    
    print(f"Model, tokenizer, and label encoder classes saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--model-path", type=str, default="./model", help="Output directory for model")
    args = parser.parse_args()
    main(args)
