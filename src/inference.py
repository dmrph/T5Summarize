import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    text = args.text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(inputs, max_length=64, min_length=20, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Summary:", summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/t5-cnn-dailymail", help="Path to the fine-tuned model")
    parser.add_argument("--text", type=str, help="Text to summarize")
    args = parser.parse_args()
    main(args)
