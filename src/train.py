import os
import yaml
import logging
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
from dataset_utils import load_cnn_dailymail_dataset, get_tokenizer, preprocess_function
from model_utils import get_model, save_model
import numpy as np

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

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1).tolist() if isinstance(predictions, np.ndarray) else predictions
    labels = labels.tolist() if isinstance(labels, np.ndarray) else labels

    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Decoding error: {e}")
        raise

    # Compute metrics
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {key: round(value.mid.fmeasure * 100, 4) for key, value in result.items()}
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return {}

def main():
    dataset = load_cnn_dailymail_dataset()
    tokenizer = get_tokenizer()

    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=512,
        num_proc=12,
        remove_columns=["article", "highlights", "id"]
    )

    if config.get("use_subset", False):
        logger.info("Using subset of the dataset for testing.")
        train_dataset = tokenized_datasets["train"].select(range(config["subset_train_size"]))
        val_dataset = tokenized_datasets["validation"].select(range(config["subset_val_size"]))
    else:
        logger.info("Using the full dataset.")
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]

    model = get_model()
    if torch.cuda.is_available():
        model = model.to("cuda")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=0.01,
        logging_steps=500,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        resume_from_checkpoint=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save model and tokenizer
    save_model(trainer, SAVE_DIR)

    # Save tokenizer explicitly
    tokenizer.save_pretrained(SAVE_DIR)
    logger.info("Training completed and model/tokenizer saved.")

if __name__ == "__main__":
    main()
