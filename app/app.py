"""
app.py

Streamlit application to input text and get a summary using the fine-tuned model.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Constants
MODEL_DIR = "models/t5-cnn-dailymail"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the fine-tuned model and tokenizer
st.title("Text Summarization Demo")
st.write("Fine-tuned T5 on CNN/DailyMail")

@st.cache_resource
def load_model_and_tokenizer():
    """
    Loads the fine-tuned model and tokenizer, and moves the model to the appropriate device.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# User Input
input_text = st.text_area("Enter text to summarize:", height=200)

if st.button("Summarize"):
    if input_text.strip():
        st.write("Processing...")
        input_ids = tokenizer.encode("summarize: " + input_text, return_tensors="pt").to(DEVICE)
        
        # Generate the summary
        outputs = model.generate(
            input_ids,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the summary
        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text.")
