"""
Heuristic layer that complements the ML model:
- red flag phrase detection
- salary sanity check
- company/email verification (rule-based, no external API)
- risk score fusion
- human-readable explanation
"""
import re

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "mail.com",
    "aol.com", "icloud.com", "protonmail.com", "yandex.com"
}

RED_FLAG_PATTERNS = [
    (r"\bno experience\b.{0,20}\bneeded\b", "Claims no experience needed for a paid role"),
    (r"\bwire transfer\b", "Mentions wire transfers (common scam payment method)"),
    (r"\bregistration fee\b|\bstarter kit\b.{0,15}\bfee\b|\bactivation fee\b", "Asks for an upfront fee"),
    (r"\bsend.{0,20}(bank details|voided check|social security)\b", "Requests sensitive financial info upfront"),
    (r"\bno interview\b|\bno resume\b.{0,15}\bneeded\b", "Skips normal hiring steps (no interview/resume)"),
    (r"\bunlimited earning\b|\bbe your own boss\b", "Uses vague 'unlimited earnings' language"),
    (r"\bact fast\b|\bapply within 24 hours\b|\blimited spots\b|\burgent hiring\b", "Uses high-pressure urgency language"),
    (r"\breship\b|\bpackage.{0,15}(receive|forward)\b", "Involves reshipping packages (classic reshipping scam)"),
    (r"\bcash paid daily\b|\bper day\b.{0,10}\$", "Unusual daily cash payment structure"),
    (r"\btext\s+\w+\s+to apply\b", "Asks to apply via text keyword (low-friction mass recruiting)"),
]

# Rough national average annual salary bands by seniority keyword, in USD.
SALARY_SANITY_BANDS = {
    "clerk": (25000, 50000), "assistant": (30000, 55000), "representative": (28000, 60000),
    "engineer": (60000, 160000), "manager": (55000, 140000), "analyst": (50000, 110000),
    "designer": (45000, 100000), "writer": (35000, 85000), "nurse": (55000, 110000),
    "accountant": (45000, 95000), "associate": (28000, 55000), "coordinator": (35000, 65000),
    "executive": (40000, 100000),
}


def extract_salary_numbers(text):
    """Pull out plausible dollar figures from free text."""
    matches = re.findall(r"\$\s?([\d,]+(?:\.\d+)?)\s*(k|K)?", text)
    values = []
    for num, k in matches:
        try:
            v = float(num.replace(",", ""))
            if k:
                v *= 1000
            values.append(v)
        except ValueError:
            continue
    return values


def analyze_salary(text):
    lower = text.lower()
    flags = []
    if re.search(r"per\s*week", lower) and re.search(r"\$\s?[2-9]\d{3}", lower):
        flags.append("Weekly pay figure implies an unrealistic annual salary")
    if re.search(r"cash paid daily|guaranteed", lower):
        flags.append("Guaranteed/daily cash pay is a common scam pattern")

    band = None
    for kw, rng in SALARY_SANITY_BANDS.items():
        if kw in lower:
            band = rng
            break

    values = extract_salary_numbers(text)
    if band and values:
        annualized = []
        for v in values:
            if re.search(r"per\s*week", lower):
                annualized.append(v * 52)
            elif re.search(r"per\s*hour|\$\s?\d+(\.\d+)?\s*/\s*hr\b|\d+\s*/\s*hr\b", lower):
                annualized.append(v * 2080)
            else:
                annualized.append(v)
        for a in annualized:
            if a > band[1] * 2.5:
                flags.append(f"Stated pay (~${a:,.0f}/yr) is far above the typical range (${band[0]:,}-${band[1]:,}) for this role")
                break
    return flags


def verify_company(company_name, email, website):
    flags = []
    score_penalty = 0
    domain = None
    if email and "@" in email:
        domain = email.split("@")[-1].strip().lower()
        if domain in FREE_EMAIL_DOMAINS:
            flags.append(f"Recruiter email uses a free/personal domain ({domain}) instead of a company domain")
            score_penalty += 25
        elif company_name:
            company_slug = re.sub(r"[^a-z0-9]", "", company_name.lower())
            domain_slug = re.sub(r"[^a-z0-9]", "", domain.split(".")[0])
            if company_slug and domain_slug and company_slug[:5] not in domain_slug and domain_slug[:5] not in company_slug:
                flags.append(f"Email domain ({domain}) doesn't obviously match company name")
                score_penalty += 10

    if not website:
        flags.append("No company website provided to verify")
        score_penalty += 10
    else:
        w = website.lower()
        if not re.match(r"^https?://", w):
            flags.append("Website URL is missing http(s):// - unable to verify format")
            score_penalty += 5

    return flags, score_penalty


def find_red_flag_phrases(text):
    lower = text.lower()
    found = []
    for pattern, explanation in RED_FLAG_PATTERNS:
        if re.search(pattern, lower):
            found.append(explanation)
    return found


def compute_risk_score(ml_fake_probability, red_flags_count, verification_penalty, salary_flags_count):
    """
    Fuses the ML model probability with heuristic signals into one 0-100 risk score.
    ML probability is weighted heaviest since it captures overall language patterns;
    heuristics catch specific known scam mechanics the model might not phrase-match on.
    """
    base = ml_fake_probability * 60  # up to 60 pts from ML
    heuristic = min(red_flags_count * 6, 20)  # up to 20 pts from phrase red flags
    salary = min(salary_flags_count * 8, 10)  # up to 10 pts from salary issues
    verification = min(verification_penalty * 0.1, 10)  # up to 10 pts from company verification
    total = base + heuristic + salary + verification
    return round(min(total, 100), 1)


def get_top_contributing_words(text, vectorizer, clf, top_n=8):
    """Returns the words/ngrams in this specific text that pushed the prediction toward 'fake'."""
    import numpy as np
    tfidf_vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    row = tfidf_vec.toarray()[0]
    contributions = row * coefs
    nonzero_idx = np.where(row > 0)[0]
    sorted_idx = nonzero_idx[np.argsort(-contributions[nonzero_idx])]
    top_words = []
    for idx in sorted_idx[:top_n]:
        if contributions[idx] > 0:
            top_words.append(feature_names[idx])
    return top_words
