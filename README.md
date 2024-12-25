# Summarization Project

This project fine-tunes a T5 model on the CNN/DailyMail dataset for text summarization, 
then serves an interactive demo via Streamlit.

## Project Structure
- `requirements.txt` for Python dependencies.
- `config/config.yaml` for hyperparameters and settings.
- `src/` folder:
  - `dataset_utils.py` handles dataset loading and preprocessing.
  - `model_utils.py` for model loading/saving helpers.
  - `train.py` for the training process.
  - `evaluate.py` for model evaluation with ROUGE.
  - `inference.py` for a quick inference script.
- `app/app.py` contains a Streamlit app for user-friendly summarization.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

2. **Train Model (optional, if you want to fine-tune yourself)**
    ```bash
    python src/train.py

3. **Run Streamlit App**
    ```bash
    streamlit run app/app.py

4. **Inference (optional, CLI approach)**:
    ```bash
    python src/inference.py --text "Your article or paragraph to summarize"
