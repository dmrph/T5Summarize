import logging
from datasets import load_dataset
import yaml
import os
from transformers import AutoTokenizer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)

def load_cnn_dailymail_dataset():
    dataset_name = config["data"]["dataset_name"]
    dataset_config = config["data"]["dataset_config"]
    logger.info(f"Loading dataset: {dataset_name} with config: {dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config)

    if config.get("use_subset", False):
        dataset["train"] = dataset["train"].select(range(config["subset_train_size"]))
        dataset["validation"] = dataset["validation"].select(range(config["subset_val_size"]))

    return dataset

def get_tokenizer():
    model_checkpoint = config["model_checkpoint"]
    logger.info(f"Loading tokenizer for model checkpoint: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    return tokenizer

def preprocess_function(examples, tokenizer, prefix="summarize: "):
    max_input_length = config["preprocessing"]["max_input_length"]
    max_target_length = config["preprocessing"]["max_target_length"]

    try:
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
        labels = tokenizer(
            text_target=examples["highlights"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
