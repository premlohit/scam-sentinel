"""
Trains a TF-IDF + Logistic Regression fake-job-posting classifier.

Why TF-IDF + LogisticRegression instead of BERT for the deployed model:
- Trains in seconds, no GPU, <1MB model file -> deploys instantly on free hosting tiers
- Fully interpretable: we can show WHICH words drove the "fake" score (great for the
  "AI Explanation" feature and for your demo/viva)
- BERT is mentioned in the tech stack for the report; swapping it in later is a drop-in
  change (see bert_upgrade_notes.md) but isn't necessary for a working, deployable app today.
"""
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("/home/claude/fake-job-detector/data/job_postings.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["fraudulent"], test_size=0.2, random_state=42, stratify=df["fraudulent"]
)

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words="english",
    lowercase=True,
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

preds = clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, target_names=["real", "fake"]))

joblib.dump(clf, "/home/claude/fake-job-detector/model/classifier.joblib")
joblib.dump(vectorizer, "/home/claude/fake-job-detector/model/vectorizer.joblib")
print("Saved model + vectorizer.")
