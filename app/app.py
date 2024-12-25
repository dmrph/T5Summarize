"""
app.py

Streamlit application to input text and get a summary using the fine-tuned model.
"""

import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = "models/t5-cnn-dailymail"

@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

def sanitize_input(text):
    """
    Sanitize input text by escaping double quotes.
    """
    return text.replace('"', '\\"')

st.title("Text Summarization Demo")
st.subheader("Fine-tuned T5 on CNN/DailyMail")

# Text input
user_input = st.text_area("Enter text to summarize:", height=200)
if st.button("Summarize"):
    if user_input.strip():
        sanitized_input = sanitize_input(user_input)
        model, tokenizer = load_model_and_tokenizer()
        inputs = tokenizer.encode("summarize: " + sanitized_input, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("### Summary")
        st.write(summary)
    else:
        st.warning("Please enter text to summarize.")
