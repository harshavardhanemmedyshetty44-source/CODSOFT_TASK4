"""
CODSOFT ML Internship - Task 4: Spam SMS Detection
------------------------------------------------------------
Classifies SMS messages as spam or legitimate (ham) using TF-IDF + Naive Bayes.
This is the simplest of the 5 tasks - do this one first if you're short on time.

DATASET: Download from the "DATASET" link in the CodSoft task PDF (Kaggle:
"SMS Spam Collection Dataset"). Typical filename: spam.csv
Columns are usually v1 (label) and v2 (message text), with some extra empty columns.

HOW TO RUN:
1. Upload spam.csv to your working folder / Colab session.
2. Run this script / paste into a Colab cell.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---- 1. Load data ----
# encoding='latin-1' avoids UnicodeDecodeError on this particular dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the two useful columns and rename them
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

print(data.head())
print(data["label"].value_counts())

# ---- 2. Encode labels: ham=0, spam=1 ----
data["label"] = data["label"].map({"ham": 0, "spam": 1})

X = data["message"]
y = data["label"]

# ---- 3. Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 4. TF-IDF vectorization ----
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ---- 5. Train classifier ----
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---- 6. Evaluate ----
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# ---- 7. Try it on custom messages ----
samples = [
    "Congratulations! You've won a free ticket to Bahamas, call now!",
    "Hey, are we still meeting for lunch tomorrow?",
]
samples_tfidf = tfidf.transform(samples)
preds = model.predict(samples_tfidf)
for msg, pred in zip(samples, preds):
    print(f"'{msg}' -> {'SPAM' if pred == 1 else 'HAM'}")
