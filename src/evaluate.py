"""
evaluate.py

Script to evaluate the trained model on the test set using ROUGE metrics.
"""

import os
import yaml
import numpy as np
import torch  # Import torch for tensor operations
from transformers import AutoModelForSeq2SeqLM
import evaluate
from dataset_utils import load_cnn_dailymail_dataset, get_tokenizer, preprocess_function

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def evaluate_model(model_dir="models/t5-summarization"):
    """
    Evaluates the trained model on the test set and computes ROUGE scores.
    """
    # Load the fine-tuned model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = get_tokenizer()

    # Load and preprocess the test dataset
    dataset = load_cnn_dailymail_dataset()
    tokenized_test = dataset["test"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["article", "highlights", "id"]
    )

    # Custom evaluation loop (recommended over a fake trainer)
    metric = evaluate.load("rouge")

    print("Running evaluation...")
    for example in tokenized_test:
        # Convert input_ids to tensors
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(input_ids.device)

        # Generate predictions
        output = model.generate(input_ids=input_ids, max_length=config["max_target_length"], num_beams=4)
        
        # Decode predictions and references
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        reference = example["highlights"]  # Adjust if your dataset uses a different key

        # Add to the metric
        metric.add(prediction=prediction, reference=reference)

    # Compute and display results
    results = metric.compute()
    print("[Test] ROUGE:", results)

if __name__ == "__main__":
    evaluate_model()
