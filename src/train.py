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
        num_proc=config["preprocessing"].get("num_proc", 4),
        remove_columns=["article", "highlights", "id"]
    )

    if config.get("use_subset", False):
        train_dataset = tokenized_datasets["train"].select(range(config["subset_train_size"]))
        val_dataset = tokenized_datasets["validation"].select(range(config["subset_val_size"]))
    else:
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]

    model = get_model()
    if torch.cuda.is_available():
        model = model.to("cuda")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=config["training"]["train_batch_size"],
        per_device_eval_batch_size=config["training"]["eval_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"],
        weight_decay=0.01,
        logging_steps=config["training"]["logging_steps"],
        fp16=True,
        load_best_model_at_end=config["training"]["resume_from_checkpoint"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        save_steps=500,
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
    save_model(trainer, SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    logger.info("Training completed and model/tokenizer saved.")
    
if __name__ == "__main__":
    main()
