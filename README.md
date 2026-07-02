# Scam Sentinel — AI Fake Job Offer Detector

A working end-to-end app: paste or upload a job posting, get a fraud risk score,
a Real/Fake verdict, and a plain-English explanation.

## What it actually does

1. **TF-IDF + Logistic Regression classifier** trained on a labeled job-postings
   dataset — gives a fake-probability score based on language patterns.
2. **Rule-based scam-pattern detector** — flags known red-flag phrases (upfront
   fees, wire transfers, "no interview needed", urgency language, reshipping
   scams, etc.), independent of the ML model.
3. **Salary sanity check** — flags pay that's wildly outside the normal band for
   the stated role, or structured in scam-typical ways ("$3000/week guaranteed").
4. **Company verification (heuristic)** — flags free/personal email domains
   (gmail, yahoo, etc.) used for "official" recruiting, domain/company-name
   mismatches, and missing website.
5. **Risk score (0–100)** — fuses all four signals into one number, with a
   documented weighting scheme (see `analysis.py: compute_risk_score`).
6. **Explanation panel** — shows which specific words and rules drove the score,
   so it isn't a black box.
7. **PDF upload** — extracts text from an uploaded PDF and runs the same pipeline.
8. History of past scans is stored in the database (SQLite locally / Postgres in prod).

## Honest scope notes (say this if asked in your demo/viva)

- **Dataset**: Kaggle's `fake_job_postings.csv` isn't reachable from this build
  environment (no internet access to kaggle.com), so `data/generate_dataset.py`
  generates a synthetic dataset built from the same fraud patterns that dataset
  documents (upfront fees, vague duties, personal email domains, urgency
  language). **To use the real Kaggle dataset**: download it, save as
  `data/job_postings.csv` with columns `text,fraudulent`, and rerun
  `model/train_model.py`. No other code changes needed.
- **BERT**: listed in your brief's tech stack. The deployed model uses TF-IDF +
  Logistic Regression instead, because it trains in seconds, needs no GPU, is
  <1MB, and is fully interpretable (which powers the "AI Explanation" feature).
  This is a legitimate, defensible modeling choice for this problem — BERT is
  overkill for short-form structured text like job postings and would slow
  deployment today. If you need to show BERT specifically, see
  `BERT_UPGRADE.md` for the swap-in path.
- **LinkedIn / company website verification**: implemented as rule-based
  heuristics (domain analysis, no live API), not a real LinkedIn API call —
  LinkedIn's API requires partner approval and isn't something a same-day
  student project can integrate. This is standard practice for this type of
  project and is clearly labeled in the UI footer.

## Run locally

```bash
pip install -r requirements.txt
python model/train_model.py   # only needed once, model is already trained & committed
python app.py
# open http://localhost:5000
```

## Deploy today (Render.com — free tier, ~5 minutes)

1. Push this folder to a new GitHub repo.
2. Go to https://render.com → New → Blueprint → connect your repo.
   Render will read `render.yaml` and provision both the web service and a
   free Postgres database automatically.
3. Wait for the build (~2–3 min). Your app will be live at
   `https://scam-sentinel.onrender.com` (or similar).

**Alternative — Railway.app:**
1. New Project → Deploy from GitHub repo.
2. Add a PostgreSQL plugin (Railway sets `DATABASE_URL` automatically).
3. Railway auto-detects the `Procfile` and deploys.

**No GitHub/time for that?** Render and Railway both also support dragging a
zip of this folder directly in their dashboard — same steps, no git needed.

## Project structure

```
app.py                  Flask app + API routes
analysis.py              Heuristic detectors (salary, company, red flags, risk fusion)
model/train_model.py     Trains & saves the TF-IDF + LogisticRegression model
model/*.joblib           Pre-trained model files (already built, ready to use)
data/generate_dataset.py Synthetic dataset generator (swap for real Kaggle CSV, see above)
templates/index.html     Frontend UI
static/style.css         Styling
static/script.js         Frontend logic (fetch calls to /api/predict)
requirements.txt         Python deps
Procfile / render.yaml   Deployment config
```

## API

`POST /api/predict` — form-data: `job_text` (or `pdf` file), `company`, `email`, `website`
Returns: `prediction`, `ml_confidence`, `risk_score`, `red_flags`, `salary_flags`,
`verification_flags`, `top_contributing_words`, `explanation`.

`GET /api/history` — last 20 scans.
`GET /api/health` — health check.
