import streamlit as st
import joblib
import numpy as np
import pandas as pd

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
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load model artifacts (saved from the notebook)
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
    "Enter the values below for each measure. This tool provides a "
    "model-based estimate and is **not a clinical diagnosis**."
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
# What the scores mean (glossary from project documentation)
# ------------------------------------------------------------
st.markdown(
    '<div style="height:4px;background:#4FD1B3;border-radius:3px;margin-bottom:2px;"></div>',
    unsafe_allow_html=True,
)
with st.expander("📋 What do these scores mean?"):
    st.markdown("""
| Measure | What it measures | Typical scale |
|---|---|---|
| Age | Student age in years | 18–56 |
| Sex | Male / Female / Other | — |
| Course Type | Undergraduate / Postgraduate / Other | — |
| Institution | Anonymised UK university code (1–6) | — |
| GAD-7 Anxiety | Generalised Anxiety Disorder Scale — anxiety level | 0–21 |
| PSS Stress | Perceived Stress Scale — overall stress | 0–40 |
| SCI Insomnia | Sleep Condition Indicator — sleep quality | 0–32 |
| UCLA-3 Loneliness | UCLA Loneliness Scale — social isolation | 20–77 |
| SBQ Suicidal Ideation | Suicidal Behaviour Questionnaire — suicidal thoughts | 3–18 |
| P16 Psychotic Exp. Sum | Prodromal Questionnaire — psychotic experiences | 0–16 |
| MDQ Mania | Mood Disorder Questionnaire — manic symptoms | 0–13 |
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
# Continuous / numeric features — documented ranges
# ------------------------------------------------------------
numeric_ranges = {
    "GAD7_Anxiety":          (0, 21, 5, 1, "GAD-7 Anxiety score (0–21)"),
    "PSS_Stress":            (0, 40, 15, 1, "Perceived Stress Scale score (0–40)"),
    "SCI_Insomnia":          (0, 32, 10, 1, "Sleep Condition Indicator score (0–32)"),
    "UCLA3_Loneliness":      (20, 77, 48, 1, "UCLA-3 Loneliness score (20–77)"),
    "SBQ_Suicidal_Ideation": (3, 18, 3, 1, "SBQ Suicidal Ideation score (3–18)"),
    "P16_Psychotic_Exp_Sum": (0, 16, 2, 1, "Psychotic experiences sum score (0–16)"),
    "MDQ_Mania":             (0, 13, 2, 1, "Mood Disorder Questionnaire score (0–13)"),
    "Age":                   (18, 56, 21, 1, "Age in years (18–56, sample mean ≈ 20.9)"),
}

# ------------------------------------------------------------
# Categorical features — dropdowns using documented codes
# ------------------------------------------------------------
SEX_OPTIONS = {"Male (1)": 1, "Female (2)": 2, "Other (3)": 3}
COURSE_OPTIONS = {
    "Undergraduate (1)": 1,
    "Postgraduate (2)": 2,
    "Other type A (3)": 3,
    "Other type B (4)": 4,
    "Other type C (5)": 5,
}
INSTITUTION_OPTIONS = {f"University {i} (anonymised)": i for i in range(1, 7)}

categorical_widgets = {
    "Sex": SEX_OPTIONS,
    "Course_Type": COURSE_OPTIONS,
    "Institution": INSTITUTION_OPTIONS,
}

categorical_help = {
    "Sex": "Biological sex as recorded in the training dataset (Male / Female / Other).",
    "Course_Type": "Type of course you are currently enrolled in (Undergraduate / Postgraduate / Other).",
    "Institution": "Anonymised university code from the training dataset — see the note below if unsure which to pick.",
}

# ------------------------------------------------------------
# Build input fields dynamically from feature_cols
# ------------------------------------------------------------
user_input = {}
st.markdown(
    '<div style="height:4px;background:#A8D4FF;border-radius:3px;margin-bottom:2px;"></div>',
    unsafe_allow_html=True,
)
with st.expander("🔗 Where to take these tests yourself"):
    st.markdown("""
Most people don't already know their GAD-7 or PSS score — these come from taking the
actual validated questionnaires. Here's where to find free, legitimate versions of each:

- **GAD-7 (Anxiety):** [ADAA official PDF](https://adaa.org/sites/default/files/2026-01/GAD-7_Anxiety-updated_0.pdf)
- **PSS-10 (Stress):** [Original Cohen scale, free PDF](https://www.ucdenver.edu/docs/librariesprovider54/wellbeing-documents/perceivedstressscale.pdf)
- **SCI (Sleep/Insomnia):** [Sleep Condition Indicator info](https://www.sleepprimarycareresources.org.au/questionnaires/sci)
- **UCLA Loneliness Scale (Version 3):** [Take it online](https://coached.com/tools/loneliness-test)
- **SBQ-R (Suicidal Ideation):** [Official PDF](https://youthsuicideprevention.nebraska.edu/wp-content/uploads/2019/09/SBQ-R.pdf)
- **MDQ (Mania):** [Official PDF](https://ibpf.org/wp-content/uploads/2016/11/MDQ.pdf)
- **P16 (Psychotic Experiences):** search "Prodromal Questionnaire-16 (PQ-16)" — mainly used in clinical/research settings, no single standard public self-test link.

Take the real test, get your score, then come back and enter it above. This app doesn't
run these tests itself — it takes scores you already have and estimates risk from them.
""")

st.subheader("Enter scores")

cols = st.columns(2)
for i, col_name in enumerate(feature_cols):
    with cols[i % 2]:
        if col_name in categorical_widgets:
            options = categorical_widgets[col_name]
            label = st.selectbox(
                col_name.replace("_", " "),
                list(options.keys()),
                help=categorical_help.get(col_name, ""),
            )
            user_input[col_name] = options[label]
        else:
            lo, hi, default, step, help_text = numeric_ranges.get(
                col_name, (0, 100, 0, 1, "")
            )
            user_input[col_name] = st.number_input(
                col_name.replace("_", " "),
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=float(step),
                help=help_text,
            )

        if col_name == "Institution":
            st.caption(
                "ℹ️ These are anonymised research codes with no public mapping to real "
                "universities — the original dataset intentionally removed that information "
                "to protect participant privacy. If you don't know which applies to you, any "
                "selection is fine; this field has a smaller effect on the prediction than "
                "measures like anxiety or stress."
            )

        if col_name == "SBQ_Suicidal_Ideation":
            st.markdown(
                '<div class="crisis-box">If you or someone you know is having '
                'thoughts of suicide, please reach out for support right away. '
                '<b>UK:</b> Samaritans — call 116 123, free, 24/7. '
                '<b>Outside the UK:</b> contact your local emergency number or a '
                'crisis line in your country.</div>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------
if st.button("Predict", type="primary"):
    X_new = pd.DataFrame([user_input])[feature_cols]
    X_new_sc = scaler.transform(X_new)

    pred = model.predict(X_new_sc)[0]
    proba = model.predict_proba(X_new_sc)[0][1]

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

st.divider()
st.caption(
    "Model: Logistic Regression | Trained on Akram et al. (2023) UK University "
    "Student Mental Health dataset (Nature Scientific Data, N=1,408) | "
    "For research/educational demonstration purposes only."
)

st.markdown(
    """
    <div style="text-align:center; padding-top: 1.2em; color:#B7C6D6; font-size:0.9em;">
        Built by <b>Irene Ufuoma Ayakazi</b> — Data Scientist &amp; Analyst<br>
        Open to freelance and remote opportunities —
        <a href="https://www.linkedin.com/in/irene-ayakazi/" target="_blank"
           style="color:#4FD1B3; text-decoration:none;">LinkedIn</a>
        &nbsp;|&nbsp;
        <a href="https://github.com/irene2024hub" target="_blank"
           style="color:#4FD1B3; text-decoration:none;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)