# Upgrading to BERT (optional, for the report/future work)

The current model (TF-IDF + Logistic Regression) is what's deployed, because
it's fast, tiny, and interpretable enough to power the explanation feature.
If you want to mention or demo a BERT-based version too:

```python
# pip install transformers torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Fine-tune on data/job_postings.csv (text, fraudulent) using Trainer API,
# then swap the predict() call in app.py to use this model instead of
# the joblib classifier. Everything else (risk fusion, heuristics, UI)
# stays the same.
```

Trade-offs to mention in your report/viva:
- BERT: higher potential accuracy on nuanced phrasing, much larger model
  (~440MB), needs more compute to train/serve, harder to explain per-word.
- TF-IDF + LogisticRegression (current): near-instant training, <1MB,
  deploys on free hosting tiers, and directly powers the "why is this fake"
  explanation via feature coefficients — which a raw BERT model doesn't give
  you for free.
