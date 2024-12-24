"""
train.py

Script to fine-tune the T5 summarization model on the CNN/DailyMail dataset.
"""

import os
import yaml
import numpy as np

import torch
from transformers import (
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer
)
import evaluate

from dataset_utils import load_cnn_dailymail_dataset, get_tokenizer, preprocess_function
from model_utils import get_model, save_model

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SAVE_DIR = config.get("save_dir", "models/t5-summarization") 

def compute_rouge(eval_pred):
    """
    Compute ROUGE scores for evaluation using the evaluate library.
    """
    metric = evaluate.load("rouge")
    tokenizer = get_tokenizer()
    
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: v.mid.fmeasure * 100 for k, v in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


def main():
    dataset = load_cnn_dailymail_dataset()
    tokenizer = get_tokenizer()

    # Preprocess
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["article", "highlights", "id"]
    )

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    model = get_model()

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        evaluation_strategy="epoch",
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=0.01,
        logging_steps=500,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_generate(model, tokenizer), 
    )

    # Train
    trainer.train()

    # Evaluate
    val_results = trainer.evaluate()
    print("[Validation] ROUGE:", val_results)

    # Save
    save_model(trainer, SAVE_DIR)


def compute_metrics_with_generate(model, tokenizer):
    """
    Returns a function for computing metrics using model's `generate` method.
    """

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        # Use the model to generate summaries
        generated_predictions = model.generate(
            torch.tensor(predictions).to(model.device),
            max_length=config["max_target_length"],
            num_beams=4,
        )

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(generated_predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Load ROUGE metric and compute scores
        metric = evaluate.load("rouge")
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {key: round(value.mid.fmeasure * 100, 4) for key, value in result.items()}

    return compute_metrics


if __name__ == "__main__":
    main()
