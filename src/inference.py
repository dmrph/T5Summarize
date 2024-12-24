"""
inference.py

A simple script to generate summaries using the fine-tuned model.
"""

import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    # Example text
    text = """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial
    intelligence concerned with the interactions between computers and human language."""
    
    if args.text:
        text = args.text
    
    # Prepend "summarize: " if the model is T5
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Summary:", summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="t5-summarization", help="Path to the fine-tuned model")
    parser.add_argument("--text", type=str, help="Custom text to summarize")
    args = parser.parse_args()
    main(args)
