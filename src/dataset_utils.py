"""
dataset_utils.py

Handles loading and preprocessing of the CNN/DailyMail dataset.
"""

import logging
from datasets import load_dataset
import yaml
import os
from transformers import AutoTokenizer

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)

def load_cnn_dailymail_dataset():
    """
    Loads the CNN/DailyMail dataset via Hugging Face 'datasets' library.
    """
    dataset_name = config["data"]["dataset_name"]
    dataset_config = config["data"]["dataset_config"]
    logger.info(f"Loading dataset: {dataset_name} with config: {dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config)
    return dataset

def get_tokenizer():
    """
    Loads the tokenizer for the specified model checkpoint in the config.
    """
    model_checkpoint = config["model_checkpoint"]
    logger.info(f"Loading tokenizer for model checkpoint: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    return tokenizer

def preprocess_function(examples, tokenizer, prefix="summarize: "):
    """
    Optimized tokenization using __call__ for T5TokenizerFast.
    """
    max_input_length = config["max_input_length"]
    max_target_length = config["max_target_length"]

    # Add prefix to articles and tokenize
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        padding="max_length",
        truncation=True
    )

    # Tokenize summaries (highlights)
    labels = tokenizer(
        examples["highlights"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
