import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
import urllib.request

TURSO_URL = st.secrets["turso"]["url"]
TURSO_TOKEN = st.secrets["turso"]["token"]

def _turso(sql, params=None):
    req_body = {"requests": [{"type": "execute", "stmt": {"sql": sql}}]}
    if params:
        args = []
        for v in params:
            if v is None:
                args.append({"type": "null", "value": None})
            elif isinstance(v, str):
                args.append({"type": "text", "value": v})
            elif isinstance(v, bool) or isinstance(v, int):
                args.append({"type": "integer", "value": str(v)})
            else:
                args.append({"type": "float", "value": v})
        req_body["requests"][0]["stmt"]["args"] = args
    payload = json.dumps(req_body).encode()
    req = urllib.request.Request(TURSO_URL + "/v2/pipeline", data=payload, method="POST")
    req.add_header("Authorization", "Bearer " + TURSO_TOKEN)
    req.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())

def save_prediction(data, pred, prob):
    sql = """INSERT INTO predictions (
        timestamp, age, sex, course_type, institution,
        sci_insomnia, gad7_anxiety, pss_stress,
        mdq_mania, sbq_suicidal_ideation,
        p16_psychotic_exp_sum, ucla3_loneliness,
        prediction, probability
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    params = (
        datetime.now().isoformat(), data["Age"], data["Sex"],
        data["Course_Type"], data["Institution"], data["SCI_Insomnia"],
        data["GAD7_Anxiety"], data["PSS_Stress"], data["MDQ_Mania"],
        data["SBQ_Suicidal_Ideation"], data["P16_Psychotic_Exp_Sum"],
        data["UCLA3_Loneliness"], int(pred), float(prob)
    )
    _turso(sql, params)

def load_stats():
    result = _turso("SELECT * FROM predictions")
    rows = result["results"][0]["response"]["result"]["rows"]
    cols = [c["name"] for c in result["results"][0]["response"]["result"]["cols"]]
    data = []
    for row in rows:
        data.append([cell.get("value") for cell in row])
    return pd.DataFrame(data, columns=cols)

st.set_page_config(
    page_title="Intelligent Depression Risk Prediction — Student Wellbeing",
    page_icon="🧠",
    layout="centered",
)

# ------------------------------------------------------------
# Professional theme (calm, clinical-but-warm palette)
# ------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, .stApp {
        background-color: #0E2A47;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 700;
    }
    p, span, label, li, div {
        color: #EAF0F6;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #B7C6D6 !important;
    }
    .stButton>button {
        background-color: #2E9E85;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.4em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #24816C;
        color: white;
    }
    div[data-testid="stExpander"] {
        background-color: #16385C;
        border-radius: 10px;
        border: 1px solid #2A4D73;
    }
    div[data-testid="stExpander"] summary {
        color: #FFFFFF !important;
    }
    div[data-testid="stNumberInput"] input {
        background-color: #16385C !important;
        color: #FFFFFF !important;
        border: 1px solid #2A4D73 !important;
        border-radius: 6px;
    }
    div[data-baseweb="select"] > div {
        background-color: #16385C !important;
        border: 1px solid #2A4D73 !important;
        border-radius: 6px;
        cursor: pointer;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #FFFFFF !important;
    }
    div[data-baseweb="popover"] li {
        background-color: #16385C !important;
        color: #FFFFFF !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #2A4D73 !important;
    }
    [data-testid="stMarkdownContainer"] table {
        color: #EAF0F6;
    }
    [data-testid="stMarkdownContainer"] table th {
        background-color: #16385C;
        color: #FFFFFF;
    }
    .crisis-box {
        background-color: #3A2B12;
        border-left: 5px solid #E0A64B;
        color: #FBEBD3;
        padding: 0.9em 1.2em;
        border-radius: 6px;
        margin-top: 0.5em;
        margin-bottom: 0.5em;
        font-size: 0.93em;
    }
    [data-baseweb="tooltip"], div[role="tooltip"] {
        background-color: #16385C !important;
        color: #FFFFFF !important;
        border: 1px solid #2A4D73 !important;
    }
    [data-baseweb="tooltip"] *, div[role="tooltip"] * {
        color: #FFFFFF !important;
    }
    hr {
        border-color: #2A4D73 !important;
    }
    .q-label {
        font-size: 0.9em;
        color: #C8DCE8 !important;
        margin-bottom: 2px;
    }
    .score-badge {
        display: inline-block;
        background-color: #2A4D73;
        color: #FFFFFF;
        padding: 0.2em 0.8em;
        border-radius: 12px;
        font-size: 0.85em;
        margin-left: 0.5em;
    }
    .section-score {
        text-align: right;
        font-size: 0.95em;
        color: #4FD1B3 !important;
        margin-top: 0.5em;
        padding: 0.3em 0.8em;
        background-color: #1A3A5C;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/best_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_cols = joblib.load("model/feature_cols.pkl")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

st.title("🧠 Intelligent Depression Risk Prediction")
st.subheader("for University Student Wellbeing")
st.write(
    "Answer the questions below for each measure. Your scores are "
    "calculated automatically. This tool provides a model-based "
    "estimate and is **not a clinical diagnosis**."
)
st.caption(
    "Model trained on Akram et al. (2023), *UK University Student Mental "
    "Health*, Nature Scientific Data (N=1,408)."
)

# ------------------------------------------------------------
# Why early detection matters
# ------------------------------------------------------------
st.markdown(
    '<div style="height:4px;background:#E0A64B;border-radius:3px;margin-bottom:2px;"></div>',
    unsafe_allow_html=True,
)
with st.expander("💡 Why early detection matters"):
    st.markdown(
        "- **Earlier support, better outcomes.** Research consistently links earlier "
        "identification of depression risk with faster access to support and reduced "
        "escalation of symptoms.\n"
        "- **Lower barrier to reaching out.** A quick, private screening can help students "
        "recognise when to talk to someone, before things feel overwhelming.\n"
        "- **Informs — doesn't replace — professional care.** A tool like this can prompt "
        "a conversation with a GP, counsellor, or university wellbeing service; it cannot "
        "diagnose or treat anything on its own.\n"
        "- **Reduces stigma through routine check-ins.** Normalising mental health screening "
        "alongside physical health checks can make seeking help feel more routine and less "
        "daunting."
    )

# ------------------------------------------------------------
# What these scores mean
# ------------------------------------------------------------
st.markdown(
    '<div style="height:4px;background:#4FD1B3;border-radius:3px;margin-bottom:2px;"></div>',
    unsafe_allow_html=True,
)
with st.expander("📋 What these scores mean"):
    st.markdown("""
Each section below corresponds to a validated psychological questionnaire. Your answers
are used to calculate a **total score** for each measure, which the prediction model
uses to estimate depression risk.

| Measure | Full Name | What it measures | Score Range |
|---|---|---|---|
| **GAD-7** | Generalised Anxiety Disorder-7 | Anxiety symptom severity over the past 2 weeks | 0–21 |
| **PSS-10** | Perceived Stress Scale-10 | How stressful you perceive your life to be over the past month | 0–40 |
| **SCI** | Sleep Condition Indicator | Sleep quality and insomnia symptoms | 0–32 |
| **UCLA-3** | UCLA Loneliness Scale (Version 3) | Subjective feelings of loneliness and social isolation | 20–80 |
| **SBQ-R** | Suicidal Behaviors Questionnaire-Revised | Suicidal ideation, communication, and likelihood | 3–18 |
| **MDQ** | Mood Disorder Questionnaire | Manic/hypomanic symptoms (bipolar spectrum screening) | 0–13 |
| **PQ-16** | Prodromal Questionnaire-16 | Psychotic-like experiences and distress | 0–64 |
""")
    st.caption(
        "Note: this app predicts a **depression risk category and probability**, not an "
        "actual PHQ-9 score. The PHQ-9 (0–27, NHS-validated) was used to train the model's "
        "labels, not as something this tool outputs directly. A result of \"Likely\" "
        "corresponds to the clinical threshold used during training (PHQ-9 ≥ 10 — moderate "
        "depressive symptoms or greater, per NHS banding) — it flags a pattern worth "
        "following up on, not a diagnosis."
    )

# ------------------------------------------------------------
# Questionnaire data structures
# ------------------------------------------------------------
FREQ_4 = [
    "Not at all (0)",
    "Several days (1)",
    "More than half the days (2)",
    "Nearly every day (3)",
]

FREQ_5 = [
    "Never (0)",
    "Almost Never (1)",
    "Sometimes (2)",
    "Fairly Often (3)",
    "Very Often (4)",
]

FREQ_UCLA = [
    "Never (1)",
    "Rarely (2)",
    "Sometimes (3)",
    "Often (4)",
]

YES_NO = ["No (0)", "Yes (1)"]

# ------------------------------------------------------------
# GAD-7
# ------------------------------------------------------------
GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen",
]

def render_gad7():
    st.markdown("#### GAD-7 — Anxiety")
    st.caption("Over the last **2 weeks**, how often have you been bothered by the following problems?")
    total = 0
    for i, q in enumerate(GAD7_QUESTIONS):
        label = f"{i+1}. {q}"
        st.markdown(f'<div class="q-label">{label}</div>', unsafe_allow_html=True)
        ans = st.selectbox(label, FREQ_4, key=f"gad7_{i}", label_visibility="collapsed")
        total += int(ans.split("(")[1].rstrip(")"))
    st.markdown(f'<div class="section-score">GAD-7 Total Score: <strong>{total}</strong> / 21</div>', unsafe_allow_html=True)
    return total

# ------------------------------------------------------------
# PSS-10
# ------------------------------------------------------------
PSS_QUESTIONS = [
    "Been upset because of something that happened unexpectedly?",
    "Felt that you were unable to control the important things in your life?",
    "Felt nervous and stressed?",
    "Felt confident about your ability to handle your personal problems?",
    "Felt that things were going your way?",
    "Found that you could not cope with all the things that you had to do?",
    "Been able to control irritations in your life?",
    "Felt that you were on top of things?",
    "Been angered because of things that happened that were outside of your control?",
    "Felt difficulties were piling up so high that you could not overcome them?",
]
PSS_REVERSE = {3, 4, 6, 7}  # 0-indexed: items 4,5,7,8

def render_pss():
    st.markdown("#### PSS-10 — Perceived Stress")
    st.caption("Over the **last month**, how often have you felt or thought the following?")
    total = 0
    for i, q in enumerate(PSS_QUESTIONS):
        st.markdown(f'<div class="q-label">{i+1}. {q}</div>', unsafe_allow_html=True)
        ans = st.selectbox(f"Q{i+1}", FREQ_5, key=f"pss_{i}", label_visibility="collapsed")
        score = int(ans.split("(")[1].rstrip(")"))
        if i in PSS_REVERSE:
            score = 4 - score
        total += score
    st.markdown(f'<div class="section-score">PSS-10 Total Score: <strong>{total}</strong> / 40</div>', unsafe_allow_html=True)
    return total

# ------------------------------------------------------------
# SCI - Sleep Condition Indicator (8 items, each 0-4)
# ------------------------------------------------------------
SCI_OPTS = {
    0: ["0–15 min (4)", "16–30 min (3)", "31–45 min (2)", "46–60 min (1)", "≥ 61 min (0)"],
    1: ["0–15 min (4)", "16–30 min (3)", "31–45 min (2)", "46–60 min (1)", "≥ 61 min (0)"],
    2: ["0–1 nights (4)", "2 nights (3)", "3 nights (2)", "4 nights (1)", "5–7 nights (0)"],
    3: ["Very good (4)", "Good (3)", "Average (2)", "Poor (1)", "Very poor (0)"],
    4: ["Not at all (4)", "A little (3)", "Somewhat (2)", "Much (1)", "Very much (0)"],
    5: ["Not at all (4)", "A little (3)", "Somewhat (2)", "Much (1)", "Very much (0)"],
    6: ["Not at all (4)", "A little (3)", "Somewhat (2)", "Much (1)", "Very much (0)"],
    7: ["No problem / < 1 mo (4)", "1–3 months (3)", "3–6 months (2)", "6–12 months (1)", "> 1 year (0)"],
}

SCI_QUESTIONS = [
    "How long does it take you to fall asleep?",
    "If you wake up during the night, how long are you awake in total?",
    "How many nights a week do you have a problem with your sleep?",
    "How would you rate your sleep quality?",
    "To what extent has poor sleep affected your mood, energy, or relationships?",
    "To what extent has poor sleep affected your concentration, productivity, or ability to stay awake?",
    "How much has poor sleep troubled you in general?",
    "How long have you had a problem with your sleep?",
]

def render_sci():
    st.markdown("#### SCI — Sleep Condition Indicator")
    st.caption("Thinking about a **typical night in the last month**...")
    total = 0
    for i, q in enumerate(SCI_QUESTIONS):
        st.markdown(f'<div class="q-label">{i+1}. {q}</div>', unsafe_allow_html=True)
        ans = st.selectbox(f"Q{i+1}", SCI_OPTS[i], key=f"sci_{i}", label_visibility="collapsed")
        total += int(ans.split("(")[1].rstrip(")"))
    st.markdown(f'<div class="section-score">SCI Total Score: <strong>{total}</strong> / 32 (higher = better sleep)</div>', unsafe_allow_html=True)
    return total

# ------------------------------------------------------------
# UCLA Loneliness Scale - Version 3 (20 items, 1-4 each)
# ------------------------------------------------------------
UCLA_QUESTIONS = [
    "How often do you feel that you are 'in tune' with the people around you?",
    "How often do you feel that you lack companionship?",
    "How often do you feel that there is no one you can turn to?",
    "How often do you feel alone?",
    "How often do you feel part of a group of friends?",
    "How often do you feel that you have a lot in common with the people around you?",
    "How often do you feel that you are no longer close to anyone?",
    "How often do you feel that your interests and ideas are not shared by those around you?",
    "How often do you feel outgoing and friendly?",
    "How often do you feel close to people?",
    "How often do you feel left out?",
    "How often do you feel that your relationships with others are not meaningful?",
    "How often do you feel that no one really knows you well?",
    "How often do you feel isolated from others?",
    "How often do you feel that you can find companionship when you want it?",
    "How often do you feel that there are people who really understand you?",
    "How often do you feel shy?",
    "How often do you feel that people are around you but not with you?",
    "How often do you feel that there are people you can talk to?",
    "How often do you feel that there are people you can turn to?",
]
UCLA_REVERSE = {0, 4, 5, 8, 9, 14, 15, 17, 18, 19}  # positively worded items

def render_ucla():
    st.markdown("#### UCLA Loneliness Scale (Version 3)")
    st.caption("Indicate how often you feel each of the following:")
    total = 0
    for i, q in enumerate(UCLA_QUESTIONS):
        st.markdown(f'<div class="q-label">{i+1}. {q}</div>', unsafe_allow_html=True)
        ans = st.selectbox(f"Q{i+1}", FREQ_UCLA, key=f"ucla_{i}", label_visibility="collapsed")
        score = int(ans.split("(")[1].rstrip(")"))
        if i in UCLA_REVERSE:
            score = 5 - score
        total += score
    st.markdown(f'<div class="section-score">UCLA Total Score: <strong>{total}</strong> / 80</div>', unsafe_allow_html=True)
    return total

# ------------------------------------------------------------
# SBQ-R (4 items, total 3-18)
# ------------------------------------------------------------
SBQ1_OPTS = [
    "Never (1)",
    "It was just a brief passing thought (2)",
    "I have had a plan at least once to kill myself but did not try to do it (3)",
    "I have had a plan at least once to kill myself and really wanted to die (4)",
    "I have attempted to kill myself, but did not want to die (5)",
    "I have attempted to kill myself, and really hoped to die (6)",
]
SBQ2_OPTS = [
    "Never (1)",
    "Rarely — 1 time (2)",
    "Sometimes — 2 times (3)",
    "Often — 3-4 times (4)",
    "Very Often — 5 or more times (5)",
]
SBQ3_OPTS = [
    "No (1)",
    "Yes, at one time, but did not really want to die (2)",
    "Yes, at one time, and really wanted to die (3)",
    "Yes, more than once, but did not want to do it (4)",
    "Yes, more than once, and really wanted to do it (5)",
]
SBQ4_OPTS = [
    "Never (0)",
    "No chance at all (1)",
    "Rather Unlikely (2)",
    "Unlikely (3)",
    "Likely (4)",
    "Rather Likely (5)",
    "Very Likely (6)",
]

def render_sbq():
    st.markdown("#### SBQ-R — Suicidal Behaviors Questionnaire (Revised)")
    st.caption("Please select the option that best applies to you.")

    st.markdown("**1. Have you ever thought about or attempted to kill yourself?**")
    s1 = int(st.selectbox("SBQ1", SBQ1_OPTS, key="sbq_1", label_visibility="collapsed").split("(")[1].rstrip(")"))

    st.markdown("**2. How often have you thought about killing yourself in the past year?**")
    s2 = int(st.selectbox("SBQ2", SBQ2_OPTS, key="sbq_2", label_visibility="collapsed").split("(")[1].rstrip(")"))

    st.markdown("**3. Have you ever told someone that you were going to commit suicide, or that you might do it?**")
    s3 = int(st.selectbox("SBQ3", SBQ3_OPTS, key="sbq_3", label_visibility="collapsed").split("(")[1].rstrip(")"))

    st.markdown("**4. How likely is it that you will attempt suicide someday?**")
    s4 = int(st.selectbox("SBQ4", SBQ4_OPTS, key="sbq_4", label_visibility="collapsed").split("(")[1].rstrip(")"))

    total = s1 + s2 + s3 + s4
    st.markdown(f'<div class="section-score">SBQ-R Total Score: <strong>{total}</strong> / 18</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="crisis-box">If you or someone you know is having '
        'thoughts of suicide, please reach out for support right away. '
        '<b>UK:</b> Samaritans — call 116 123, free, 24/7. '
        '<b>Outside the UK:</b> contact your local emergency number or a '
        'crisis line in your country.</div>',
        unsafe_allow_html=True,
    )
    return total

# ------------------------------------------------------------
# MDQ - Mood Disorder Questionnaire (13 yes/no items)
# ------------------------------------------------------------
MDQ_QUESTIONS = [
    "Felt so good or so hyper that other people thought you were not your normal self, or you were so hyper that you got into trouble?",
    "Been so irritable that you shouted at people or started fights or arguments?",
    "Felt much more self-confident than usual?",
    "Got much less sleep than usual and found you didn't really miss it?",
    "Been much more talkative or spoken faster than usual?",
    "Had thoughts racing through your head, or couldn't slow your mind down?",
    "Been so easily distracted that you had trouble concentrating or staying on track?",
    "Had much more energy than usual?",
    "Been much more active or done many more things than usual?",
    "Been much more social or outgoing than usual (e.g., telephoned friends in the middle of the night)?",
    "Been much more interested in sex than usual?",
    "Done things that were unusual or that others might have thought excessive, foolish, or risky?",
    "Had spending money that got you or your family in trouble?",
]

def render_mdq():
    st.markdown("#### MDQ — Mood Disorder Questionnaire")
    st.caption("**Has there ever been a period of time when you were not your usual self and…**")
    total = 0
    for i, q in enumerate(MDQ_QUESTIONS):
        st.markdown(f'<div class="q-label">{i+1}. {q}</div>', unsafe_allow_html=True)
        ans = st.selectbox(f"Q{i+1}", YES_NO, key=f"mdq_{i}", label_visibility="collapsed")
        total += int(ans.split("(")[1].rstrip(")"))
    st.markdown(f'<div class="section-score">MDQ Total Score: <strong>{total}</strong> / 13</div>', unsafe_allow_html=True)
    return total

# ------------------------------------------------------------
# PQ-16 - Prodromal Questionnaire (16 items, True/False + distress)
# ------------------------------------------------------------
P16_QUESTIONS = [
    "I feel uninterested in the things I used to enjoy",
    "I often experience strange feelings (e.g., déjà vu) with no explanation",
    "I sometimes smell or taste things that others do not notice",
    "I often hear unusual sounds like ringing or hissing when there is nothing there",
    "I sometimes see things that others do not see (shadows, flashes, shapes)",
    "I often feel that others are watching or talking about me",
    "I sometimes feel that I am being singled out or that there is a special message for me",
    "I feel that others do not understand me or cannot relate to me",
    "I sometimes feel that my thoughts are not my own or are being controlled",
    "I feel that I have trouble communicating clearly",
    "I often feel suspicious or mistrustful of others",
    "I sometimes feel that something strange or unexplainable is happening",
    "I often feel nervous or anxious in social situations",
    "I sometimes feel that people are plotting against me",
    "I feel that my mind is 'empty' or that I cannot think clearly",
    "I sometimes feel disconnected from my own body or surroundings",
]
P16_OPTS = [
    "False (0)",
    "True, No distress (1)",
    "True, Mild distress (2)",
    "True, Moderate distress (3)",
    "True, Severe distress (4)",
]

def render_p16():
    st.markdown("#### PQ-16 — Prodromal Questionnaire")
    st.caption("For each statement, indicate if it is true or false, and if true, how distressing it is.")
    total = 0
    for i, q in enumerate(P16_QUESTIONS):
        st.markdown(f'<div class="q-label">{i+1}. {q}</div>', unsafe_allow_html=True)
        ans = st.selectbox(f"Q{i+1}", P16_OPTS, key=f"p16_{i}", label_visibility="collapsed")
        val = int(ans.split("(")[1].rstrip(")"))
        total += val
    st.markdown(f'<div class="section-score">PQ-16 Total Score: <strong>{total}</strong> / 64</div>', unsafe_allow_html=True)
    return total

# ------------------------------------------------------------
# Build input fields
# ------------------------------------------------------------
st.markdown(
    '<div style="height:4px;background:#A8D4FF;border-radius:3px;margin-bottom:2px;"></div>',
    unsafe_allow_html=True,
)

st.subheader("📋 Questionnaires")

# Demographics section
st.markdown("#### About You")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=56, value=21, step=1, help="Your age in years")
with col2:
    sex_opts = {"Male (1)": 1, "Female (2)": 2, "Other (3)": 3}
    sex_label = st.selectbox("Sex", list(sex_opts.keys()))
    sex = sex_opts[sex_label]
with col3:
    course_opts = {
        "Undergraduate (1)": 1,
        "Postgraduate (2)": 2,
        "Other type A (3)": 3,
        "Other type B (4)": 4,
        "Other type C (5)": 5,
    }
    course_label = st.selectbox("Course Type", list(course_opts.keys()))
    course = course_opts[course_label]

inst_opts = {f"University {i} (anonymised)": i for i in range(1, 7)}
inst_label = st.selectbox("Institution", list(inst_opts.keys()))
institution = inst_opts[inst_label]

st.caption(
    "ℹ️ Institution codes are anonymised research codes with no public mapping to real "
    "universities. If unsure, any selection is fine — this field has a smaller effect "
    "on the prediction than measures like anxiety or stress."
)

# Questionnaire sections in expanders
with st.expander("😰 GAD-7 — Anxiety (7 questions)", expanded=False):
    gad7 = render_gad7()

with st.expander("😣 PSS-10 — Perceived Stress (10 questions)", expanded=False):
    pss = render_pss()

with st.expander("😴 SCI — Sleep Condition Indicator (8 questions)", expanded=False):
    sci = render_sci()

with st.expander("👤 UCLA Loneliness Scale (20 questions)", expanded=False):
    ucla = render_ucla()

with st.expander("⚠️ SBQ-R — Suicidal Behaviours (4 questions)", expanded=False):
    sbq = render_sbq()

with st.expander("📈 MDQ — Mood Disorder (13 questions)", expanded=False):
    mdq = render_mdq()

with st.expander("🧠 PQ-16 — Prodromal Experiences (16 questions)", expanded=False):
    p16 = render_p16()

# Collect all feature values in the order expected by the model
user_input = {
    "Age": age,
    "Sex": sex,
    "Course_Type": course,
    "SCI_Insomnia": sci,
    "GAD7_Anxiety": gad7,
    "PSS_Stress": pss,
    "MDQ_Mania": mdq,
    "SBQ_Suicidal_Ideation": sbq,
    "P16_Psychotic_Exp_Sum": p16,
    "UCLA3_Loneliness": ucla,
    "Institution": institution,
}

# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------
st.markdown(
    '<div style="height:4px;background:#4FD1B3;border-radius:3px;margin-bottom:2px;"></div>',
    unsafe_allow_html=True,
)

if st.button("Predict", type="primary"):
    X_new = pd.DataFrame([user_input])[feature_cols]
    X_new_sc = scaler.transform(X_new)

    pred = model.predict(X_new_sc)[0]
    proba = model.predict_proba(X_new_sc)[0][1]

    save_prediction(user_input, pred, proba)

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Depression risk: **Likely** (probability: {proba*100:.1f}%)")
        st.markdown(
            '<div class="crisis-box">This result suggests it may help to talk to '
            'someone soon — a GP, your university\'s wellbeing/counselling service, '
            'or a trusted person in your life. If you are in crisis, in the UK you '
            'can call Samaritans free on 116 123, any time. If you are outside the '
            'UK, please contact your local emergency services or a crisis helpline '
            'in your country.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.success(f"✅ Depression risk: **Unlikely** (probability: {proba*100:.1f}%)")

    st.caption(
        "This is a statistical estimate based on a machine learning model trained on "
        "questionnaire data. It is not a diagnostic tool. If you or someone you know is "
        "struggling, please consult a mental health professional."
    )

# ------------------------------------------------------------
# Average Data
# ------------------------------------------------------------
with st.expander("📊 Average Data"):
    df = load_stats()
    if df.empty:
        st.caption("No predictions recorded yet.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Predictions", len(df))
        pct = df["prediction"].mean() * 100
        m2.metric("Depression Likely", f"{pct:.1f}%")
        m3.metric("Depression Unlikely", f"{100-pct:.1f}%")

        st.markdown("**Average Scores Across All Users:**")
        score_cols = ["age", "sex", "course_type", "institution",
                      "sci_insomnia", "gad7_anxiety", "pss_stress",
                      "mdq_mania", "sbq_suicidal_ideation",
                      "p16_psychotic_exp_sum", "ucla3_loneliness"]
        labels = ["Age", "Sex", "Course Type", "Institution",
                  "SCI (Insomnia)", "GAD-7 (Anxiety)", "PSS (Stress)",
                  "MDQ (Mania)", "SBQ-R (Suicidal Ideation)",
                  "PQ-16 (Psychotic Exp.)", "UCLA-3 (Loneliness)"]
        avg = df[score_cols].mean().round(1)
        avg_df = pd.DataFrame({"Measure": labels, "Average Score": avg.values})
        st.dataframe(avg_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Model: Logistic Regression | Trained on Akram et al. (2023) UK University "
    "Student Mental Health dataset (Nature Scientific Data, N=1,408) | "
    "For research/educational demonstration purposes only."
)

st.markdown(
    """
    <div style="text-align:center; padding-top: 1.2em; color:#B7C6D6; font-size:0.9em;">
        Built by <b style="color:#FFD700;">Irene Ufuoma Ayakazi</b> — Data Scientist &amp; Analyst<br>
        Open to freelance and remote opportunities —
        <a href="https://www.linkedin.com/in/irene-ayakazi/" target="_blank"
           style="color:#4A90D9; text-decoration:none;">LinkedIn</a>
        &nbsp;|&nbsp;
        <a href="https://github.com/irene2024hub" target="_blank"
           style="color:#4A90D9; text-decoration:none;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
