import os
import io
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import joblib
from pypdf import PdfReader

from analysis import (
    analyze_salary, verify_company, find_red_flag_phrases,
    compute_risk_score, get_top_contributing_words
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Works with SQLite locally; on Render/Railway/Heroku set DATABASE_URL to a
# Postgres connection string (they all provide one automatically) and the
# app switches over with zero code changes.
db_url = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'app.db')}")
if db_url.startswith("postgres://"):  # SQLAlchemy needs postgresql://
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    company = db.Column(db.String(255))
    prediction = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    text_snippet = db.Column(db.Text)


with app.app_context():
    db.create_all()

# Load ML model + vectorizer once at startup
clf = joblib.load(os.path.join(BASE_DIR, "model", "classifier.joblib"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.joblib"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    text = ""
    company = request.form.get("company", "")
    email = request.form.get("email", "")
    website = request.form.get("website", "")

    if "pdf" in request.files and request.files["pdf"].filename:
        pdf_file = request.files["pdf"]
        try:
            reader = PdfReader(io.BytesIO(pdf_file.read()))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            return jsonify({"error": f"Could not read PDF: {e}"}), 400
    else:
        text = request.form.get("job_text", "") or (request.json.get("job_text") if request.is_json else "")

    if not text or not text.strip():
        return jsonify({"error": "No job description text provided."}), 400

    # ML prediction
    tfidf_vec = vectorizer.transform([text])
    proba = clf.predict_proba(tfidf_vec)[0]
    fake_probability = float(proba[1])
    ml_prediction = "Fake" if fake_probability >= 0.5 else "Real"

    # Heuristics
    red_flags = find_red_flag_phrases(text)
    salary_flags = analyze_salary(text)
    verification_flags, verification_penalty = verify_company(company, email, website)

    risk_score = compute_risk_score(
        ml_fake_probability=fake_probability,
        red_flags_count=len(red_flags),
        verification_penalty=verification_penalty,
        salary_flags_count=len(salary_flags),
    )

    final_prediction = "Fake" if risk_score >= 50 else "Real"
    top_words = get_top_contributing_words(text, vectorizer, clf) if final_prediction == "Fake" else []

    all_flags = red_flags + salary_flags + verification_flags
    explanation = build_explanation(final_prediction, risk_score, top_words, all_flags)

    record = Prediction(
        company=company or "N/A",
        prediction=final_prediction,
        confidence=round(fake_probability if final_prediction == "Fake" else 1 - fake_probability, 3),
        risk_score=risk_score,
        text_snippet=text[:300],
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "prediction": final_prediction,
        "ml_confidence": round(fake_probability * 100, 1),
        "risk_score": risk_score,
        "red_flags": red_flags,
        "salary_flags": salary_flags,
        "verification_flags": verification_flags,
        "top_contributing_words": top_words,
        "explanation": explanation,
    })


def build_explanation(prediction, risk_score, top_words, flags):
    if prediction == "Fake":
        parts = [f"This posting scored {risk_score}/100 on our risk scale, indicating it is likely FAKE."]
        if top_words:
            parts.append("Language patterns most associated with fraud: " + ", ".join(top_words) + ".")
        if flags:
            parts.append("Specific concerns detected: " + "; ".join(flags[:5]) + ".")
        parts.append("Recommendation: do not share personal/financial information, and verify the company independently before proceeding.")
    else:
        parts = [f"This posting scored {risk_score}/100 on our risk scale, indicating it is likely REAL."]
        if flags:
            parts.append("Minor points worth double-checking: " + "; ".join(flags[:3]) + ".")
        else:
            parts.append("No major red flags were detected in the text or provided company details.")
        parts.append("Recommendation: still verify the company independently before sharing sensitive information, as with any job application.")
    return " ".join(parts)


@app.route("/api/history", methods=["GET"])
def history():
    records = Prediction.query.order_by(Prediction.created_at.desc()).limit(20).all()
    return jsonify([{
        "company": r.company,
        "prediction": r.prediction,
        "confidence": r.confidence,
        "risk_score": r.risk_score,
        "created_at": r.created_at.isoformat(),
    } for r in records])


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
