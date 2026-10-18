# Gujarati–English Code-Mixed Text Understanding System

A Flask web app demonstrating:
- Word-level language identification (starter lexicons/rules)
- Informal spelling normalization
- Dictionary-based Romanized Gujarati transliteration
- Sentiment, intent and topic classification with character TF-IDF + calibrated Linear SVM

## Requirements
- Python 3.10+ recommended
- Internet is not required to run after packages are installed.
- This baseline does not download a transformer model. Transformer embeddings can be added later.

## Run on Windows (VS Code)

Open the project folder in VS Code, then run these commands in the terminal:

```powershell
py -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

The first run trains three small models and saves them under `models/`. If you change the CSV dataset, delete the three `.pkl` files in `models/` and restart the app to retrain.

## Dataset
`data/code_mixed_dataset.csv` is a small starter dataset for demonstrating the pipeline. It is intentionally small and should not be used to claim high accuracy. For a college evaluation, collect and manually label more diverse examples, including Gujarati script, Romanized Gujarati, English, Hindi and mixed sentences.

Columns:
- `text`
- `sentiment`: Positive / Negative / Neutral
- `intent`: Question / Request / Complaint / Appreciation / Statement
- `topic`: Education / Shopping / Technology / Entertainment / Weather / General

## Limitations and academic honesty
- Word language tags use small dictionaries and simple rules; unknown or ambiguous words can be mislabeled.
- Transliteration is dictionary-based and does not transliterate arbitrary Romanized Gujarati perfectly.
- The starter training data is illustrative. The output is a prototype prediction, not a validated model.
- For a stronger research project, add human-labeled data, evaluate with held-out test data and macro-F1, and compare this baseline with XLM-RoBERTa or another multilingual transformer.

## Project structure
```text
gujarati_code_mixed_nlp/
├── app.py
├── nlp_utils.py
├── requirements.txt
├── README.md
├── data/
│   └── code_mixed_dataset.csv
├── models/                 # created on first run
├── templates/
│   └── index.html
└── static/
    └── style.css
```
