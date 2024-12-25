import os
import yaml
from transformers import AutoModelForSeq2SeqLM

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def get_model():
    """
    Loads a Seq2Seq model (e.g., T5) from the model_checkpoint in the config.
    """
    model_checkpoint = config["model_checkpoint"]
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return model

def save_model(trainer, output_dir="t5-summarization"):
    """
    Saves the trained model & tokenizer to the output directory.
    """
    trainer.save_model(output_dir)
    if trainer.tokenizer:
        trainer.tokenizer.save_pretrained(output_dir)
    else:
        print("Warning: Tokenizer was not found in the trainer object!")
