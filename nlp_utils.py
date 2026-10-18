import os
import re
import csv
import pickle
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "code_mixed_dataset.csv"
MODEL_DIR = ROOT / "models"

# Small starter lexicons. Expand these using reviewed examples from your target domain.
GUJARATI_ROMAN = {
    "hu", "hun", "mane", "maru", "mari", "maro", "tame", "tamne", "ame",
    "aapde", "aa", "aaje", "kal", "kem", "shu", "su", "chhe", "che", "nathi",
    "bahu", "saras", "majama", "mastu", "sachu", "khub", "pan", "ane", "ma",
    "mate", "sathe", "thi", "par", "javu", "karvu", "karo", "karo", "karyu",
    "thayu", "thase", "joie", "jo", "nathi", "game", "gamyु", "avse", "aavse",
    "moklo", "mokli", "aapo", "aapjo", "khabar", "samaj", "samjayu", "kemke",
    "chhokra", "chhokri", "ghar", "dost", "mitra", "paisa", "sarु", "saru",
    "kharekhar", "have", "pachi", "pahela", "to", "j", "pan", "ek", "nathi"
}
HINDI_ROMAN = {
    "hai", "hain", "mujhe", "mera", "meri", "mere", "aap", "tum", "kya",
    "kyun", "kaise", "bahut", "achha", "accha", "nahi", "nahin", "chahiye",
    "kripya", "dhanyavad", "shukriya", "matlab", "lekin", "aur", "se", "ko",
    "ka", "ki", "ke", "mein", "me", "ho", "tha", "thi", "the", "karna"
}
ENGLISH_COMMON = set("""
i me my we our you your he she it they them this that these those is are am was
were be been being have has had do does did can could will would should may might
must please send share need want like love hate good bad very not no yes what when
where why how which who product movie assignment exam date class college teacher
student order delivery refund help issue problem complaint thank thanks today
tomorrow weather nice amazing excellent terrible poor service quality price
buy purchase learn code python website app technology education shopping
entertainment request question working works working project is are was were
""".split())

NORMALIZATION_MAP = {
    "plz": "please", "pls": "please", "thnx": "thanks", "thx": "thanks",
    "gud": "good", "gr8": "great", "u": "you", "ur": "your",
    "nhi": "nathi", "nai": "nathi", "sarasuu": "saras", "saras": "saras",
    "bahuu": "bahu", "cheee": "che", "chheee": "chhe"
}

# Explicit transliteration examples improve usability for common Romanized phrases.
ROMAN_TO_GUJARATI = {
    "hu": "હું", "mane": "મને", "maru": "મારું", "mari": "મારી", "tame": "તમે",
    "aa": "આ", "aaje": "આજે", "shu": "શું", "su": "શું", "kem": "કેમ",
    "chhe": "છે", "che": "છે", "nathi": "નથી", "bahu": "બહુ", "saras": "સરસ",
    "ane": "અને", "pan": "પણ", "ma": "માં", "mate": "માટે", "sathe": "સાથે",
    "thi": "થી", "moklo": "મોકલો", "aapo": "આપો", "joie": "જોઈએ",
    "game": "ગમે", "gamyુ": "ગમ્યું", "saru": "સારું", "majama": "મજામાં",
    "aavse": "આવશે", "avse": "આવશે", "thayu": "થયું", "karvu": "કરવું",
    "karo": "કરો", "kemke": "કેમકે", "have": "હવે", "pachi": "પછી",
    "ghar": "ઘર", "dost": "દોસ્ત", "khabar": "ખબર", "samjayu": "સમજાયું",
    "maro": "મારો", "tamne": "તમને", "aapde": "આપણે", "javu": "જવું"
}

def tokenize(text):
    # Keep Gujarati Unicode words, Latin words/numbers, and punctuation.
    return re.findall(r"[\u0A80-\u0AFF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text, flags=re.UNICODE)

def normalize_text(text):
    text = text.strip()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # coooool -> cool
    tokens = tokenize(text)
    normalized = []
    for token in tokens:
        key = token.lower()
        normalized.append(NORMALIZATION_MAP.get(key, token))
    return " ".join(normalized)

def identify_language(token):
    low = token.lower()
    if re.fullmatch(r"[\u0A80-\u0AFF]+", token):
        return "Gujarati (Gujarati script)"
    if not re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", token):
        return "Punctuation/Number"
    if low in GUJARATI_ROMAN:
        return "Gujarati (Romanized)"
    if low in HINDI_ROMAN:
        return "Hindi (Romanized)"
    if low in ENGLISH_COMMON:
        return "English"
    # Unknown Latin tokens are marked as uncertain rather than confidently guessing.
    return "English/Unknown"

def transliterate_to_gujarati(text):
    out = []
    for token in tokenize(text):
        low = token.lower()
        if low in ROMAN_TO_GUJARATI:
            out.append(ROMAN_TO_GUJARATI[low])
        elif re.fullmatch(r"[A-Za-z]+", token) and identify_language(token) == "Gujarati (Romanized)":
            # Unknown Romanized Gujarati: preserve it rather than invent a conversion.
            out.append(token)
        else:
            out.append(token)
    return " ".join(out)

def _train_one(df, target):
    # Character n-grams are helpful for spelling variation and code-mixed text.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                                  sublinear_tf=True, max_features=30000)),
        ("clf", CalibratedClassifierCV(LinearSVC(), cv=3))
    ])
    model.fit(df["text"].astype(str), df[target].astype(str))
    return model

def ensure_models():
    MODEL_DIR.mkdir(exist_ok=True)
    needed = ["sentiment", "intent", "topic"]
    if all((MODEL_DIR / f"{name}.pkl").exists() for name in needed):
        return
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", *needed])
    for target in needed:
        # Small data is for a demonstrator; each label should have >=3 examples for cv=3.
        model = _train_one(df, target)
        with open(MODEL_DIR / f"{target}.pkl", "wb") as f:
            pickle.dump(model, f)

def analyze_text(text):
    ensure_models()
    normalized = normalize_text(text)
    tokens = tokenize(text)
    word_labels = [
        {"word": token, "language": identify_language(token)}
        for token in tokens
        if token.strip()
    ]
    predictions = {}
    for target in ["sentiment", "intent", "topic"]:
        with open(MODEL_DIR / f"{target}.pkl", "rb") as f:
            model = pickle.load(f)
        predictions[target] = str(model.predict([normalized])[0]).title()
    return {
        "original": text,
        "normalized": normalized,
        "transliteration": transliterate_to_gujarati(text),
        "word_labels": word_labels,
        "sentiment": predictions["sentiment"],
        "intent": predictions["intent"],
        "topic": predictions["topic"],
    }
