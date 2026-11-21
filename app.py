"""Loan Approval Predictor - Flask app for Render.

Model: scikit-learn DecisionTreeClassifier (class 1 = approved, class 0 = rejected).
Place your model file next to this script as `decision.pkl` (or set MODEL_PATH).
"""
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "decision.pkl"))

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

# The pickle holds integer-encoded categories (alphabetical, LabelEncoder style).
GENDER = {"female": 0, "male": 1}
EDUCATION = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
HOME = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
INTENT = {
    "DEBTCONSOLIDATION": 0,
    "EDUCATION": 1,
    "HOMEIMPROVEMENT": 2,
    "MEDICAL": 3,
    "PERSONAL": 4,
    "VENTURE": 5,
}
DEFAULTS = {"No": 0, "Yes": 1}

app = Flask(__name__)

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoanLens · Loan Approval Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0b1020; --card:rgba(255,255,255,.06); --border:rgba(255,255,255,.12);
    --text:#e8ecf7; --muted:#97a0bd; --accent:#7c5cff; --accent2:#22d3ee;
    --good:#22c55e; --bad:#ef4444;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Inter',system-ui,sans-serif;color:var(--text);min-height:100vh;
    background:
      radial-gradient(900px 500px at 10% -10%,rgba(124,92,255,.35),transparent 60%),
      radial-gradient(800px 500px at 100% 0%,rgba(34,211,238,.25),transparent 60%),
      var(--bg);
    padding:32px 16px 60px;
  }
  .wrap{max-width:1080px;margin:0 auto}
  header{text-align:center;margin-bottom:34px}
  .logo{display:inline-flex;align-items:center;gap:10px;font-weight:800;font-size:1.05rem;
        padding:6px 14px;border:1px solid var(--border);border-radius:999px;background:var(--card)}
  .logo i{width:10px;height:10px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--accent2))}
  h1{font-size:clamp(2rem,5vw,3.2rem);font-weight:800;margin:18px 0 10px;letter-spacing:-.03em;
     background:linear-gradient(90deg,#fff,#b7a9ff 50%,var(--accent2));-webkit-background-clip:text;
     background-clip:text;color:transparent}
  header p{color:var(--muted);max-width:560px;margin:0 auto;line-height:1.6}
  .grid{display:grid;grid-template-columns:1.5fr 1fr;gap:22px;align-items:start}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:26px;
        backdrop-filter:blur(14px);box-shadow:0 20px 60px rgba(0,0,0,.35)}
  .card h2{font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
           margin:0 0 16px;font-weight:700}
  .card h2:not(:first-child){margin-top:26px}
  .fields{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
  @media(max-width:560px){.fields{grid-template-columns:1fr}}
  label{display:block;font-size:.8rem;color:var(--muted);margin-bottom:6px;font-weight:500}
  input,select{width:100%;background:rgba(0,0,0,.28);border:1px solid var(--border);color:var(--text);
        border-radius:12px;padding:12px 14px;font:inherit;font-size:.95rem;outline:none;transition:.2s}
  input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(124,92,255,.25)}
  select option{background:#151b33}
  button{margin-top:26px;width:100%;border:0;cursor:pointer;color:#fff;font:inherit;font-weight:700;
         font-size:1rem;padding:15px;border-radius:14px;
         background:linear-gradient(135deg,var(--accent),#5b8cff 55%,var(--accent2));
         box-shadow:0 10px 30px rgba(124,92,255,.4);transition:.2s}
  button:hover{transform:translateY(-2px)}
  button:disabled{opacity:.6;cursor:wait;transform:none}
  .result{position:sticky;top:20px;text-align:center}
  .placeholder{color:var(--muted);padding:40px 10px;line-height:1.7}
  .placeholder b{display:block;font-size:2.6rem;margin-bottom:8px}
  .gauge{width:200px;height:200px;margin:6px auto 18px;position:relative}
  .gauge svg{transform:rotate(-90deg)}
  .gauge .num{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
  .gauge .num strong{font-size:2.6rem;font-weight:800}
  .gauge .num span{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
  .badge{display:inline-block;padding:9px 22px;border-radius:999px;font-weight:800;font-size:1.05rem;letter-spacing:.02em}
  .badge.ok{background:rgba(34,197,94,.15);color:var(--good);border:1px solid rgba(34,197,94,.45)}
  .badge.no{background:rgba(239,68,68,.15);color:var(--bad);border:1px solid rgba(239,68,68,.45)}
  .note{margin-top:18px;color:var(--muted);font-size:.85rem;line-height:1.6}
  .stats{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:20px;text-align:left}
  .stat{background:rgba(0,0,0,.25);border:1px solid var(--border);border-radius:12px;padding:10px 12px}
  .stat span{display:block;font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
  .stat b{font-size:1rem}
  .err{color:var(--bad);margin-top:14px;font-size:.9rem}
  footer{text-align:center;color:var(--muted);font-size:.78rem;margin-top:34px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo"><i></i> LoanLens</div>
    <h1>Loan Approval Predictor</h1>
    <p>Enter applicant and loan details. A decision-tree model trained on historical loan data estimates whether the application would be approved.</p>
  </header>

  <div class="grid">
    <form class="card" id="form">
      <h2>Applicant</h2>
      <div class="fields">
        <div><label>Age</label><input type="number" name="person_age" min="18" max="100" value="30" required></div>
        <div><label>Gender</label>
          <select name="person_gender">{% for k in gender %}<option>{{k}}</option>{% endfor %}</select></div>
        <div><label>Education</label>
          <select name="person_education">{% for k in education %}<option {{'selected' if k=='Bachelor'}}>{{k}}</option>{% endfor %}</select></div>
        <div><label>Work experience (years)</label><input type="number" name="person_emp_exp" min="0" max="80" value="5" required></div>
        <div><label>Annual income ($)</label><input type="number" name="person_income" min="0" step="100" value="70000" required></div>
        <div><label>Home ownership</label>
          <select name="person_home_ownership">{% for k in home %}<option {{'selected' if k=='RENT'}}>{{k}}</option>{% endfor %}</select></div>
      </div>

      <h2>Loan</h2>
      <div class="fields">
        <div><label>Loan amount ($)</label><input type="number" name="loan_amnt" min="100" step="100" value="10000" required></div>
        <div><label>Loan purpose</label>
          <select name="loan_intent">{% for k in intent %}<option {{'selected' if k=='PERSONAL'}}>{{k}}</option>{% endfor %}</select></div>
        <div><label>Interest rate (%)</label><input type="number" name="loan_int_rate" min="0" max="40" step="0.01" value="11.0" required></div>
        <div><label>Loan as % of income</label><input type="text" id="pct" disabled></div>
      </div>

      <h2>Credit</h2>
      <div class="fields">
        <div><label>Credit score</label><input type="number" name="credit_score" min="300" max="850" value="680" required></div>
        <div><label>Credit history (years)</label><input type="number" name="cb_person_cred_hist_length" min="0" max="60" value="6" required></div>
        <div style="grid-column:1/-1"><label>Previous loan defaults on file?</label>
          <select name="previous_loan_defaults_on_file">{% for k in defaults %}<option>{{k}}</option>{% endfor %}</select></div>
      </div>

      <button type="submit" id="btn">Check eligibility</button>
      <div class="err" id="err"></div>
    </form>

    <div class="card result" id="result">
      <div class="placeholder"><b>📊</b>Fill in the form and press <strong>Check eligibility</strong> to see the prediction.</div>
    </div>
  </div>
  <footer>For educational use only — not financial advice or a real credit decision.</footer>
</div>

<script>
const form=document.getElementById('form'), res=document.getElementById('result');
const pct=document.getElementById('pct');
function upd(){const i=+form.person_income.value,l=+form.loan_amnt.value;
  pct.value=i>0?((l/i)*100).toFixed(1)+' %':'—';}
form.addEventListener('input',upd);upd();

form.addEventListener('submit',async e=>{
  e.preventDefault();
  const btn=document.getElementById('btn'),err=document.getElementById('err');
  err.textContent='';btn.disabled=true;btn.textContent='Analyzing…';
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(Object.fromEntries(new FormData(form)))});
    const d=await r.json(); if(!r.ok) throw new Error(d.error||'Request failed');
    show(d);
  }catch(x){err.textContent=x.message}
  btn.disabled=false;btn.textContent='Check eligibility';
});

function show(d){
  const p=Math.round(d.approval_probability*100), ok=d.approved;
  const col=ok?'#22c55e':'#ef4444', C=2*Math.PI*84;
  res.innerHTML=`
    <h2>Prediction</h2>
    <div class="gauge">
      <svg width="200" height="200" viewBox="0 0 200 200">
        <circle cx="100" cy="100" r="84" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="14"/>
        <circle id="arc" cx="100" cy="100" r="84" fill="none" stroke="${col}" stroke-width="14"
          stroke-linecap="round" stroke-dasharray="${C}" stroke-dashoffset="${C}"
          style="transition:stroke-dashoffset 1s ease"/>
      </svg>
      <div class="num"><strong>${p}%</strong><span>approval score</span></div>
    </div>
    <div class="badge ${ok?'ok':'no'}">${ok?'✓ Likely approved':'✕ Likely rejected'}</div>
    <div class="stats">
      <div class="stat"><span>Loan amount</span><b>$${d.summary.loan_amnt.toLocaleString()}</b></div>
      <div class="stat"><span>Loan / income</span><b>${d.summary.pct}%</b></div>
      <div class="stat"><span>Credit score</span><b>${d.summary.credit_score}</b></div>
      <div class="stat"><span>Interest rate</span><b>${d.summary.rate}%</b></div>
    </div>
    <p class="note">${d.note}</p>`;
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    document.getElementById('arc').style.strokeDashoffset=C*(1-p/100);}));
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        PAGE,
        gender=GENDER,
        education=EDUCATION,
        home=HOME,
        intent=INTENT,
        defaults=DEFAULTS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json(force=True)
        income = float(d["person_income"])
        loan = float(d["loan_amnt"])
        if income <= 0 or loan <= 0:
            raise ValueError("Income and loan amount must be greater than 0.")
        pct = loan / income

        row = {
            "person_age": float(d["person_age"]),
            "person_gender": GENDER[d["person_gender"]],
            "person_education": EDUCATION[d["person_education"]],
            "person_income": income,
            "person_emp_exp": float(d["person_emp_exp"]),
            "person_home_ownership": HOME[d["person_home_ownership"]],
            "loan_amnt": loan,
            "loan_intent": INTENT[d["loan_intent"]],
            "loan_int_rate": float(d["loan_int_rate"]),
            "loan_percent_income": round(pct, 2),
            "cb_person_cred_hist_length": float(d["cb_person_cred_hist_length"]),
            "credit_score": float(d["credit_score"]),
            "previous_loan_defaults_on_file": DEFAULTS[d["previous_loan_defaults_on_file"]],
        }
        X = pd.DataFrame([row])[FEATURES]
        proba = float(model.predict_proba(X)[0][1])
        approved = bool(model.predict(X)[0] == 1)

        if approved:
            note = "The model sees a profile similar to previously approved applicants."
        elif row["previous_loan_defaults_on_file"] == 1:
            note = "A previous default on file is the strongest signal against approval."
        else:
            note = "Lower loan-to-income ratio, a lower rate or a higher credit score may help."

        return jsonify(
            approved=approved,
            approval_probability=proba,
            note=note,
            summary=dict(
                loan_amnt=int(loan),
                pct=round(pct * 100, 1),
                credit_score=int(row["credit_score"]),
                rate=row["loan_int_rate"],
            ),
        )
    except (KeyError, ValueError, TypeError) as e:
        return jsonify(error=f"Invalid input: {e}"), 400


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
