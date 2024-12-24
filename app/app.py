"""
app.py

Streamlit application to input text and get a summary using the fine-tuned model.
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load fine-tuned model and tokenizer
# Adjust path if needed
MODEL_DIR = "t5-summarization"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

st.title("Text Summarization Demo")
st.write("Fine-tuned T5 on CNN/DailyMail")

input_text = st.text_area("Enter text to summarize:", height=200)

if st.button("Summarize"):
    if input_text.strip():
        input_ids = tokenizer.encode("summarize: " + input_text, return_tensors="pt")
        outputs = model.generate(
            input_ids,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text.")
