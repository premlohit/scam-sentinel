"""
Generates a synthetic fake/real job postings dataset modeled on the
patterns documented in the Kaggle "Real or Fake Job Posting" dataset
(vague duties, upfront-fee requests, unrealistic pay, personal email
domains, urgency language, poor grammar, etc.)

To use the REAL Kaggle dataset instead: download fake_job_postings.csv
from Kaggle, place it in this folder, and point train_model.py at it
(it expects columns: text, fraudulent). This script is a drop-in
replacement so the rest of the pipeline doesn't change.
"""
import pandas as pd
import random

random.seed(42)

titles = [
    "Data Entry Clerk", "Software Engineer", "Customer Service Representative",
    "Marketing Manager", "Administrative Assistant", "Financial Analyst",
    "Remote Chat Support Agent", "Product Manager", "Graphic Designer",
    "Sales Executive", "Human Resources Coordinator", "Content Writer",
    "Virtual Personal Assistant", "Accountant", "Warehouse Associate",
    "Mystery Shopper", "Payment Processing Agent", "Social Media Manager",
    "Registered Nurse", "Business Development Representative",
]

companies_real = [
    "Northbridge Analytics", "Vertex Logistics Inc", "BlueRiver Health Systems",
    "Cascade Manufacturing Co", "Summit Retail Group", "Anchor Financial Partners",
    "Clearwater Software", "Pioneer Consulting", "Redwood Legal Services",
    "Harborview Media",
]

companies_fake = [
    "Global Earnings Hub", "QuickCash Solutions LLC", "WorkFromHome Elite",
    "PrimeJobs Direct", "EasyMoney Careers", "Remote Wealth Partners",
    "Instant Hire Co", "TopDollar Jobs Network", "FreedomWork Agency",
    "SwiftPay Employment",
]

real_duty_bank = [
    "Collaborate with cross-functional teams to deliver quarterly project milestones.",
    "Prepare monthly financial reports and reconcile departmental budgets.",
    "Respond to customer inquiries via phone and email within SLA targets.",
    "Maintain accurate records in the internal CRM and update pipeline status weekly.",
    "Coordinate interview scheduling and onboarding paperwork for new hires.",
    "Design wireframes and prototypes in Figma based on stakeholder feedback.",
    "Conduct quality assurance testing and document defects in Jira.",
    "Draft and edit marketing copy for the company blog and email newsletters.",
    "Assist the finance team with accounts payable and expense reconciliation.",
    "Operate warehouse equipment safely in accordance with OSHA guidelines.",
    "Analyze sales data and present findings to the regional director monthly.",
    "Support patients and coordinate care plans with attending physicians.",
]

fake_duty_bank = [
    "Earn $500-$1000 per day working just 2 hours from home, no experience needed!",
    "We will mail you a check to deposit and you keep a percentage as payment.",
    "No interview required, immediate hire, just fill out the form below.",
    "You will process payments and forward a portion via wire transfer.",
    "Simply reply with your bank details so we can set up direct deposit for training pay.",
    "Unlimited earning potential, be your own boss, work whenever you want!",
    "A small registration fee of $49 is required to activate your employee starter kit.",
    "This is a golden opportunity to make thousands weekly with zero skills required.",
    "We need someone to receive packages at home and reship them internationally.",
    "Act fast, only 3 positions left, apply within 24 hours to secure your spot.",
    "Send a copy of your ID and a voided check to begin the hiring process today.",
    "Congratulations, you have been selected, no resume review necessary.",
]

real_email_domains = ["company.com", "corp.com", "careers-company.com"]
fake_email_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "mail.com"]

locations = ["New York, NY", "Austin, TX", "Chicago, IL", "Remote", "Seattle, WA", "Boston, MA"]

def make_real(i):
    title = random.choice(titles)
    company = random.choice(companies_real)
    duties = " ".join(random.sample(real_duty_bank, k=3))
    salary = random.choice(["$55,000 - $70,000 per year", "$28/hr", "$65,000 annually", "Competitive, based on experience"])
    loc = random.choice(locations)
    email = f"hr@{random.choice(real_email_domains)}"
    text = (
        f"Job Title: {title}\nCompany: {company}\nLocation: {loc}\n"
        f"Salary: {salary}\nContact: {email}\n\n"
        f"About the role: {company} is hiring a {title} to join our growing team. "
        f"{duties} We offer a comprehensive benefits package including health insurance, "
        f"401k matching, and paid time off. Candidates should have relevant experience "
        f"and submit a resume and cover letter through our careers portal."
    )
    return text, 0

def make_fake(i):
    title = random.choice(titles)
    company = random.choice(companies_fake)
    duties = " ".join(random.sample(fake_duty_bank, k=3))
    salary = random.choice(["$3000-$5000 per WEEK guaranteed!!!", "$150,000/yr no experience needed", "Cash paid daily"])
    loc = random.choice(["Remote", "Work From Anywhere", "Anywhere in the US"])
    email = f"hiring.{random.randint(100,999)}@{random.choice(fake_email_domains)}"
    text = (
        f"Job Title: {title}\nCompany: {company}\nLocation: {loc}\n"
        f"Salary: {salary}\nContact: {email}\n\n"
        f"URGENT HIRING!!! {company} is looking for motivated individuals to start immediately. "
        f"{duties} No resume needed, no interview, start earning today! "
        f"Text WORK to apply now, limited spots available."
    )
    return text, 1

rows = []
for i in range(300):
    rows.append(make_real(i))
for i in range(300):
    rows.append(make_fake(i))

random.shuffle(rows)
df = pd.DataFrame(rows, columns=["text", "fraudulent"])
df.to_csv("/home/claude/fake-job-detector/data/job_postings.csv", index=False)
print(df.shape)
print(df["fraudulent"].value_counts())
