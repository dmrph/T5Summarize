"""
evaluate.py

Script to evaluate the trained model on the test set using ROUGE metrics.
"""

import os
import yaml
import torch
import logging
from transformers import AutoModelForSeq2SeqLM
import evaluate
from dataset_utils import load_cnn_dailymail_dataset, get_tokenizer, preprocess_function

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def evaluate_model(model_dir="models/t5-cnn-dailymail"):
    """
    Evaluates the trained model on the test set and computes ROUGE scores.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")
    tokenizer = get_tokenizer()

    dataset = load_cnn_dailymail_dataset()
    tokenized_test = dataset["test"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=128,
        num_proc=12,  # Utilize CPU threads
        remove_columns=["article", "highlights", "id"]
    )

    metric = evaluate.load("rouge")
    logger.info("Starting evaluation...")
    for example in tokenized_test:
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to("cuda")
        output = model.generate(input_ids=input_ids, max_length=config["max_target_length"], num_beams=4)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        reference = example["highlights"]
        metric.add(prediction=prediction, reference=reference)

    results = metric.compute()
    logger.info(f"[Test] ROUGE: {results}")


if __name__ == "__main__":
    evaluate_model()
