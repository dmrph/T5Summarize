"""
train.py

Script to fine-tune the T5 summarization model on the CNN/DailyMail dataset.
"""

import os
import yaml
import torch
import logging
from transformers import (
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from dataset_utils import load_cnn_dailymail_dataset, get_tokenizer, preprocess_function
from model_utils import get_model, save_model

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SAVE_DIR = config.get("save_dir", "models/t5-cnn-dailymail")


def compute_metrics(eval_pred):
    """
    Compute ROUGE metrics during evaluation.
    """
    predictions, labels = eval_pred
    tokenizer = get_tokenizer()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Load ROUGE metric
    from datasets import load_metric
    metric = load_metric("rouge")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Format results
    return {key: round(value.mid.fmeasure * 100, 4) for key, value in result.items()}


def main():
    dataset = load_cnn_dailymail_dataset()
    tokenizer = get_tokenizer()

    # Preprocess with optimized tokenization
    logger.info("Starting tokenization...")
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=128,  # Adjusted for RAM
        num_proc=12,     # Fully utilize CPU threads
        remove_columns=["article", "highlights", "id"]
    )

    # Save tokenized datasets for reuse
    tokenized_datasets.save_to_disk("data/processed/cnn_dailymail")
    logger.info("Tokenization completed and saved.")

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    # Load model
    model = get_model()
    if torch.cuda.is_available():
        model = model.to("cuda")

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
        save_total_limit=2
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save model
    save_model(trainer, SAVE_DIR)
    logger.info("Model training completed and saved.")


if __name__ == "__main__":
    main()
