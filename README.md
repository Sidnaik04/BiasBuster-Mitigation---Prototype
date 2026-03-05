# BiasBuster – Bias Mitigation Prototype

BiasBuster is a research prototype for detecting and mitigating bias in machine learning models.
It provides an end-to-end fairness workflow:

- Upload dataset and model
- Run baseline fairness audit
- Get mitigation strategy recommendation
- Apply mitigation (SMOTE, Reweighting, Threshold Optimization)
- Compare fairness/performance before vs after
- Rank mitigation strategies automatically
- Review results in a Streamlit dashboard

---

## System Flow

```text
Dataset + Model Upload
        ↓
Baseline Fairness Audit
        ↓
Strategy Recommendation
        ↓
Mitigation Execution
(SMOTE / Reweighting / Threshold)
        ↓
Fairness Comparison
        ↓
Automatic Strategy Ranking
        ↓
Debiased Model Artifact
        ↓
Streamlit Dashboard
```

---

## Project Structure

```text
mitigation_prototype/
├── app/
│   ├── api/
│   │   ├── upload.py
│   │   ├── baseline.py
│   │   ├── mitigation.py
│   │   └── auto_mitigation.py
│   ├── core/
│   │   ├── dataset_loader.py
│   │   ├── model_loader.py
│   │   ├── preprocessing.py
│   │   └── persistence.py
│   ├── fairness/
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── mitigation/
│   │   ├── smote.py
│   │   ├── reweighting.py
│   │   ├── threshold.py
│   │   ├── recommender.py
│   │   └── strategy_ranker.py
│   ├── db/
│   │   ├── database.py
│   │   └── models.py
│   ├── artifacts/
│   │   ├── datasets/
│   │   └── models/
│   └── main.py
├── streamlit_app/
│   ├── app.py
│   ├── api_client.py
│   └── charts.py
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Setup

### 1) Clone

```bash
git clone https://github.com/Sidnaik04/BiasBuster-Mitigation---Prototype.git
cd biasbuster/mitigation_prototype
```

### 2) Create and activate virtual environment

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

### Start FastAPI backend

```bash
uvicorn app.main:app --reload
```

- API: http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

### Start Streamlit frontend

In a second terminal:

```bash
streamlit run streamlit_app/app.py
```

- Dashboard: http://localhost:8501

---

## How to Test

### 1) Upload dataset and model

- Dataset: CSV file
- Model: serialized scikit-learn model (`.pkl` or `.joblib`)

### 2) Select columns

- Target column (example: `income`)
- Sensitive attribute (example: `gender`, `age`, or `race`)

### 3) Run baseline audit

Computed metrics include:

- Performance: Accuracy, Precision, Recall, F1
- Fairness: Selection Rate, DPD, DIR, EOD

### 4) Get recommendation

The system recommends one of:

- SMOTE
- Reweighting
- Threshold Optimization

### 5) Run mitigation

The system retrains/applies mitigation and returns:

- Before vs after fairness/performance metrics
- Comparison summary
- Saved mitigated model artifact path

### 6) Run automatic strategy ranking

After running all three strategies, the system ranks them based on fairness gain and performance trade-off, and returns the best strategy.

---

## API Endpoints

- `POST /upload/`
- `POST /baseline/`
- `POST /mitigation/recommend`
- `POST /mitigation/apply`
- `POST /auto-mitigation/rank`
- `GET /health`

---

## Recommended Test Datasets

- Adult Income (sensitive attributes: gender, race)
- COMPAS (sensitive attribute: race)
- German Credit (sensitive attribute: age)

---

## Current Scope

- Bias analysis for tabular CSV datasets
- Primarily classification-focused workflows
- Mitigation via SMOTE, reweighting, and thresholding

---

## Contributors

- Sid Naik
- <add teammate names>

---

## License

This project is intended for research and educational purposes.
