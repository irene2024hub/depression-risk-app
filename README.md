# Intelligent Depression Risk Prediction for University Student Wellbeing

A Logistic Regression model that estimates depression risk in university
students from validated mental health questionnaire scores, deployed as a
Streamlit web app in Docker.

## Dataset
Akram, U. et al. (2023). *UK University Student Mental Health* [Dataset].
Nature Scientific Data. N = 1,408 UK university students.

> Note: your training pipeline used 1,126 samples (an 80/20-style train
> split of the full N=1,408 dataset). Worth stating clearly in your
> report/writeup that 1,126 refers to the training subset, not the full
> dataset.

## 1. Export the model from your notebook
Run `save_artifacts_RUN_IN_NOTEBOOK.py` (copy its contents into a notebook cell)
so it creates:
```
model/best_model.pkl
model/scaler.pkl
model/feature_cols.pkl
```
Then move the `model/` folder into this project folder (next to `app.py`).

## 2. Folder structure should look like this
```
depression-app/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── model/
    ├── best_model.pkl
    ├── scaler.pkl
    └── feature_cols.pkl
```

## 3. Build the Docker image
```bash
docker build -t depression-predictor .
```

## 4. Run the container
```bash
docker run -p 8501:8501 depression-predictor
```

## 5. Open the app
Visit **http://localhost:8501** in your browser.

## Questionnaires & Scoring Reference

The model's input features come from validated mental health screening
instruments. This app does **not** reproduce the exact question wording
(these instruments are copyrighted by their original authors/publishers) —
use the official versions linked below when building your data-collection
form, and cite them accordingly.

| Feature | Instrument | What it measures | Typical range | Official source |
|---|---|---|---|---|
| `PHQ9_Depression` | PHQ-9 (Patient Health Questionnaire) | Depression severity, 9 items | 0–27 | Pfizer / [phqscreeners.com](https://www.phqscreeners.com) |
| `GAD7_Anxiety` | GAD-7 (Generalized Anxiety Disorder scale) | Anxiety severity, 7 items | 0–21 | Pfizer / [phqscreeners.com](https://www.phqscreeners.com) |
| `PSS_Stress` | PSS (Perceived Stress Scale) | Perceived stress, 10-item version | 0–40 | Cohen, Kamarck & Mermelstein (1983) |
| `SCI_Insomnia` | SCI (Sleep Condition Indicator) | Insomnia symptoms | 0–28 | Espie et al. (2014) |
| `UCLA3_Loneliness` | UCLA-3 (3-item Loneliness Scale) | Loneliness | 3–9 | Hughes et al. (2004) |
| `SBQ_Suicidal_Ideation` | SBQ-R (Suicidal Behaviors Questionnaire) | Suicidal ideation/behavior risk | 0–18 | Osman et al. (2001) |
| `P16_Psychotic_Exp_Sum` | PQ-16 / P16 | Psychotic-like experiences | 0–16 | Ising et al. (2012) |
| `MDQ_Mania` | MDQ (Mood Disorder Questionnaire) | Mania/hypomania screening | 0–13 | Hirschfeld et al. (2000) |
| `Age` | — | Participant age (years) | Numeric | — |
| `Institution` | — | Institution code as encoded in the source dataset | Categorical | Akram et al. (2023) |
| `Sex` | — | Sex as encoded in the source dataset | Categorical | Akram et al. (2023) |
| `Course_Type` | — | Course/programme type code | Categorical | Akram et al. (2023) |

**Important:** the exact scoring, item wording, and administration instructions
for each clinical instrument should be obtained from the official source
before deploying this to real users — do not write your own version of the
questions. For `Institution`, `Sex`, and `Course_Type`, use the exact
categorical encoding scheme from the Akram et al. (2023) dataset documentation
so inputs map correctly to what the model was trained on.

⚠️ **Ethical/clinical note:** several of these instruments (particularly the
PHQ-9 and SBQ-R) touch on suicidal ideation. If this app is used with real
students, it should sit behind appropriate safeguarding protocols — e.g. a
visible crisis-support resource, and a clear pathway to a human counselor for
anyone flagged as high-risk — rather than being used as a standalone,
unsupervised tool.

## Notes
- Edit the `ranges` dictionary in `app.py` to match the real min/max/scale of
  each questionnaire (GAD-7, PSS, etc.) and how categorical fields
  (Institution, Sex, Course_Type) were encoded during training.
- To deploy publicly (not just locally), push the image to a registry
  (Docker Hub, AWS ECR, etc.) and run it on a cloud VM, or use a platform
  like Render, Railway, or AWS App Runner that deploys directly from a
  Dockerfile.
- This app displays a disclaimer since it's a mental-health-related
  prediction tool — keep that in place for any real-world use.
