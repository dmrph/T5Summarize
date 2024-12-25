from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_dir = "models/t5-cnn-dailymail"
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
