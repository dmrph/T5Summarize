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
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = get_tokenizer()

        if torch.cuda.is_available():
            model = model.to("cuda")

        dataset = load_cnn_dailymail_dataset()
        tokenized_test = dataset["test"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            batch_size=32,
            remove_columns=["article", "highlights", "id"]
        )

        rouge = evaluate.load("rouge")
        logger.info("Starting evaluation...")

        results = []
        for example in tokenized_test:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to("cuda")
            output = model.generate(input_ids=input_ids, max_length=config["max_target_length"], num_beams=4)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            reference = example["highlights"]
            results.append({"prediction": prediction, "reference": reference})

        metrics = rouge.compute(predictions=[r["prediction"] for r in results],
                                references=[r["reference"] for r in results])
        logger.info(f"[Test] ROUGE Scores: {metrics}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    evaluate_model()
