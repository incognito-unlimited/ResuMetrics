import os
import argparse
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import nltk
import transformers.integrations

nltk.download('punkt')

def main(args):
    # Debug: Print training data path
    print(f"Loading training data from: {args.training_data}")
    
    # Load data
    df = pd.read_csv(args.training_data)
    
    # Combine resume_text and job_description (optional)
    df['input_text'] = df['resume_text']  # Use resume_text only
    # Uncomment to include job_description:
    # df['input_text'] = df['resume_text'] + " [SEP] " + df['job_description']
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df[['input_text', 'label']])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["input_text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    # Split dataset
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2  # Binary classification (adjust if needed)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=os.path.join(args.output_dir, "logs"),
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
    
    # Determine output path for saving model (prefer AZUREML_OUTPUT_DIR)
    output_path = os.environ.get("AZUREML_OUTPUT_DIR", args.output_dir)
    print(f"Saving model and tokenizer to: {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save model and tokenizer explicitly to output_path
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Model and tokenizer saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a resume matching model")
    parser.add_argument("--training-data", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--output-dir", type=str, default="./model", help="Output directory for model")
    args = parser.parse_args()
    main(args)
