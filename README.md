# pruefomat – Fraud & Verification Assistant

Welcome to **pruefomat**, a human-in-the-loop fraud triage toolkit. This guide is written for complete beginners and walks you from zero to a working workflow, including training, interactive analysis, batch scoring, and feedback collection.

---

## 1. What You Need

### Hardware & OS
- macOS, Windows, or Linux with at least 8 GB RAM.
- 5 GB free disk space for virtual environments, models, and reports.

### Software Prerequisites
1. **Python 3.11** (recommended).
2. **Git** (to clone or update the repository).
3. **Virtual environment tool** (conda or `python -m venv`).
4. **Optional**: Visual Studio Code or another editor for viewing reports/notebooks.

> **Tip:** If you are unsure about Python versions, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and create a dedicated environment.

---

## 2. Quick Installation

```bash
# clone the project
git clone <your-repo-url> pruefomat
cd pruefomat

# create & activate a virtual environment (using venv)
python -m venv .venv
source .venv/bin/activate   # on macOS/Linux
# .venv\Scripts\activate   # on Windows PowerShell

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Conda users:** `conda env create -f environment.yml` if you maintain a separate definition.

---

## 3. Folder Tour

- `app.py` – Gradio user interface (training, analysis, feedback, batch scoring).
- `build_pipeline_veri.py` – legacy CLI for prototyping with the Veri dataset.
- `src/train_binary.py` – binary fraud triage trainer (CV, metrics, SHAP, reports).
- `src/eval_utils.py` – helper functions for metrics and visualisations.
- `config/` – default configuration and schema files.
- `tests/` – pytest coverage for core functionality.
- `reports/` – generated metrics, plots, SHAP summaries, Markdown reports.
- `feedback.db` – SQLite database for human feedback (auto-created).

---

## 4. Training a Binary Fraud Model (CLI)

Run the trainer on an Excel export with the schema defined in `src/data_schema.yaml`:

```bash
python -m src.train_binary \
  --excel data/Veri-Bsp.xlsx \
  --sheet 0 \
  --config src/data_schema.yaml \
  --outdir reports/B1 \
  --kfolds 5 \
  --topk "0.05,0.10,0.20" \
  --random_state 42
```

The script produces:
- `reports/B1/metrics.json` – aggregated CV metrics (ROC-AUC, PR-AUC, precision@k, lift, costs).
- `reports/B1/pr_curve.png`, `precision_at_k.png`, `lift.png` – visual diagnostics.
- `reports/B1/feature_importance.png`, `shap_summary.png`, `shap_examples.png` – explainability.
- `reports/B1/preprocessor.joblib`, `reports/B1/model.joblib` – ready-to-use pipeline + calibrated classifier.
- `reports/B1/predictions.csv` – ranked predictions across the full dataset.

If you pass `--predictions-out some/path.csv` the trainer writes predictions there as well.

---

## 5. Using the Gradio App

Launch the interactive app:

```bash
python app.py
```

Open the printed URL (e.g. `http://127.0.0.1:7860`) in your browser. You will see two tabs.

### Tab 1 – Training & Analyse
1. **Daten laden** – upload an Excel file and optionally a YAML config.
2. **Pipeline bauen** – creates preprocessing pipeline (numeric, categorical, text).
3. **Features Vorschau** – inspect transformed features (first rows).
4. **Baseline trainieren** – fits model, calibration, metrics; stores:
   - `predictions.csv`
   - feature importance, SHAP background data
5. **Erklärung anzeigen** – provide a row index (0-based) to view SHAP top-5 features.
6. **Pattern Report generieren** – Markdown summary (feature importances, fraud rate by Land/BUK, text keywords).
7. **Feedback** – enter index/user/comment and label (**✅ True Positive** or **❌ False Positive**). Feedback entries go into `feedback.db`.
8. **Feedback-Report** – summarises last-week precision and totals, returns Markdown.

All downloads appear as clickable files below each action.

### Tab 2 – Batch Prediction
- Upload a large Excel file (no labels required).
- Click **Belege prüfen**. The app shows a progress bar and produces an Excel download with added `fraud_score` (0–100) and `prediction` columns.

> **Performance note:** Batch inference for ~1000 rows completes in a few seconds on a laptop.

---

## 6. Feedback & Human-in-the-Loop

All verdicts recorded via the UI are stored in `feedback.db` (SQLite) with fields:
```
(beleg_index, beleg_id, timestamp, user, score, prediction, feedback, comment)
```
Use `feedback_report_action` in the UI to track precision last week. For retraining:
1. Periodically export `SELECT * FROM feedback` as CSV.
2. Merge new labels into your training dataset.
3. Re-run `src/train_binary.py` or integrate into an automated pipeline once you have >100 new labels.

---

## 7. Running Tests & Coverage

```bash
pytest --cov=. --cov-report=html
```

Open `htmlcov/index.html` to review module-level coverage. The test suite exercises parsing utilities, pipeline construction, Gradio callbacks (patched for SHAP), batch processing, and feedback recording.

To integrate with GitHub Actions, create `.github/workflows/tests.yml` referencing the same command.

---

## 8. Troubleshooting

| Issue | Fix |
| ----- | --- |
| `ModuleNotFoundError` after install | Ensure virtual environment active (`source .venv/bin/activate`). |
| Excel parsing errors | Check sheet name/index, ensure columns match schema. |
| SHAP too slow | In testing we patch SHAP; for production consider caching and background jobs. |
| Feedback DB locked | Close other processes accessing `feedback.db`, or change path via `PRUEFOMAT_SETTINGS`. |
| Browser blocked localhost | Use `app.launch(share=True)` in `app.py` if needed. |

---

## 9. Configuration Tips

- Default column types live in `src/data_schema.yaml`. Adjust categorical/text lists to suit your data.
- For custom models edit `prepare_classifiers` in `src/train_binary.py`.
- Batch export can be redirected by setting environment variable:
  ```bash
  export PRUEFOMAT_SETTINGS='{"feedback_db": "~/pruefomat/feedback.db"}'
  ```

---

## 10. Roadmap / Next Steps

- Automate weekly retraining pipeline.
- Add additional cost models and threshold optimisation.
- Integrate scheduler (cron, Airflow) for reports and feedback ingestion.
- Harden SHAP processing for very large vocabularies (sampling strategies).

---

Happy auditing! If you get stuck, open an issue with logs (`run.log`/console output) or contact the team.
