"""
Fake News Detection (single file)

Install:  pip install pandas scikit-learn
Run:      python fake_news_detection.py

Agar data/Fake.csv aur data/True.csv (Kaggle ISOT dataset) maujood hain
to unse train hoga. Nahi hain to chhote built-in sample se chalega.
"""
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FAKE_PATH = "data/Fake.csv"
TRUE_PATH = "data/True.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"^.{0,60}?\(reuters\)\s*-\s*", "", text)  # source tag hatao
    text = re.sub(r"http\S+|www\.\S+", " ", text)            # links hatao
    text = re.sub(r"[^a-z\s]", " ", text)                    # sirf letters
    return re.sub(r"\s+", " ", text).strip()


def sample_data():
    fake = [
        "Shocking secret miracle cure doctors do not want you to know",
        "Celebrity reveals aliens control the government in leaked video",
        "You will not believe what this politician did, media is hiding it",
        "Drinking this one drink melts belly fat overnight guaranteed",
        "Breaking: moon landing admitted to be fake by insider",
        "Government puts mind control chemicals in tap water, exposed",
        "Scientists baffled as man lives 200 years eating only this fruit",
        "Anonymous source says election was rigged by secret machines",
        "Share before it is deleted: vaccine turns people into zombies",
        "Free iPhone for everyone, click this link now to claim",
    ]
    real = [
        "Parliament passes budget bill after two days of debate",
        "Central bank keeps interest rates unchanged amid inflation concerns",
        "Researchers publish study on climate trends in peer reviewed journal",
        "Local council approves new bridge construction project",
        "Prime minister meets foreign leaders to discuss trade agreement",
        "Stock markets close higher as technology shares gain",
        "Health ministry reports decline in seasonal flu cases this month",
        "University announces new scholarship program for students",
        "Court hears arguments in long running property dispute case",
        "Railway department announces new train services between cities",
    ]
    return pd.DataFrame({
        "content": [clean_text(t) for t in fake + real],
        "label": [0] * len(fake) + [1] * len(real),
    })


def load_data():
    if os.path.exists(FAKE_PATH) and os.path.exists(TRUE_PATH):
        print("Kaggle dataset mil gaya, usse train ho raha hai...")
        fake = pd.read_csv(FAKE_PATH)
        true = pd.read_csv(TRUE_PATH)
        fake["label"], true["label"] = 0, 1
        df = pd.concat([fake, true], ignore_index=True)
        df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean_text)
        return df[df["content"].str.len() > 0].drop_duplicates(subset="content")
    print("Dataset nahi mila, chhote sample data se train ho raha hai.")
    print("(Achhe result ke liye Kaggle se Fake.csv aur True.csv data/ folder me rakho)\n")
    return sample_data()


def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.25,
        random_state=42, stratify=df["label"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=5)),
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"],
                                zero_division=0))
    return model


def predict(model, text):
    proba = model.predict_proba([clean_text(text)])[0]
    label = "REAL" if proba[1] >= 0.5 else "FAKE"
    return label, max(proba)


if __name__ == "__main__":
    model = train_model()
    print("News paste karo (band karne ke liye 'quit' likho)\n")
    while True:
        text = input("News: ").strip()
        if text.lower() == "quit":
            break
        if not text:
            continue
        label, conf = predict(model, text)
        print(f"Result: {label} ({conf:.1%} confidence)\n")
