"""
Fraud-Selector Prototype v0.1
---------------------------------
A minimal end-to-end demo for selecting review candidates with a controllable
randomness (epsilon) using the Credit Card Fraud dataset schema.

Run locally:
  pip install gradio pandas scikit-learn matplotlib numpy
  python app.py

Then open the local URL printed in the console.

Notes:
- Works with the common Kaggle/UCI credit card fraud CSV (columns: Time, V1..V28, Amount, Class)
- Explanations here are limited to feature importances since V1..V28 are PCA components.
- Later, swap the feature engineering + column names to your invoice schema with minimal changes.
"""

import io
import os
import numpy as np
import pandas as pd
import gradio as gr
from dataclasses import dataclass
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

@dataclass
class TrainResult:
    model: Pipeline
    ap: float
    pr_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]
    threshold_at_k: float
    feature_names: list


def load_csv(file_obj) -> pd.DataFrame:
    if isinstance(file_obj, str):
        df = pd.read_csv(file_obj)
    else:
        # gradio provides tempfile-like objects
        df = pd.read_csv(file_obj.name)
    # Basic sanity: expected columns present
    expected_last = {"Amount", "Class"}
    if not expected_last.issubset(set(df.columns)):
        raise ValueError("CSV must include 'Amount' and 'Class' columns (plus V1..V28 and Time).")
    return df


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["Class"])  # use all other cols as features
    y = df["Class"].astype(int)
    return X, y


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> TrainResult:
    X, y = split_xy(df)

    # time-based split could be used; for simplicity we use stratified random split here
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pipeline: scale -> logistic regression -> calibration (isotonic)
    base = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # sparse-safe
        ("logreg", LogisticRegression(max_iter=200, class_weight="balanced")),
    ])
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)

    clf.fit(X_train, y_train)

    # Evaluate AP (PR-AUC)
    proba = clf.predict_proba(X_test)[:, 1]
    ap = float(average_precision_score(y_test, proba))

    precision, recall, thresholds = precision_recall_curve(y_test, proba)

    # Pick a default K ~= 2% of test set as plausible daily review budget proportion
    k_ratio = 0.02
    k = max(1, int(len(y_test) * k_ratio))

    # threshold at top-k
    topk_idx = np.argsort(proba)[::-1][:k]
    thr_at_k = float(sorted(proba, reverse=True)[k-1])

    return TrainResult(model=clf, ap=ap, pr_curve=(precision, recall, thresholds),
                       threshold_at_k=thr_at_k, feature_names=list(X.columns))


def plot_pr_curve(pr_curve: Tuple[np.ndarray, np.ndarray, np.ndarray], ap: float) -> np.ndarray:
    precision, recall, _ = pr_curve
    plt.figure(figsize=(5,4))
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (AP={ap:.3f})")
    plt.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=144)
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def select_candidates(df: pd.DataFrame, model: Pipeline, budget: int, epsilon: float, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = split_xy(df)
    scores = model.predict_proba(X)[:, 1]
    df_out = df.copy()
    df_out["score"] = scores

    # Deterministic top part
    n_top = max(0, int(round(budget * (1 - epsilon))))
    n_rand = max(0, budget - n_top)

    # Get top-n by score
    top_idx = np.argsort(scores)[::-1][:n_top]

    # Random exploration among the remainder
    remaining_idx = np.setdiff1d(np.arange(len(df_out)), top_idx)
    rand_idx = rng.choice(remaining_idx, size=min(n_rand, len(remaining_idx)), replace=False)

    chosen = np.concatenate([top_idx, rand_idx])
    selection = df_out.iloc[chosen].copy()

    # Log propensity: simple mixture model prop (not exact but indicative)
    base_prop = epsilon * (1.0 / max(1, len(remaining_idx)))
    selection["propensity"] = np.where(
        selection.index.isin(top_idx),
        (1 - epsilon),
        base_prop,
    )

    # Nice ordering: top first, then random by score
    selection.sort_values(by=["score"], ascending=False, inplace=True)

    # Keep a compact view
    cols = [c for c in df.columns if c not in ("Class",)] + ["score", "propensity", "Class"]
    return selection[cols]


def quick_importance(df: pd.DataFrame, model: Pipeline, n: int = 10) -> pd.DataFrame:
    # fallback: use absolute LR coefficients after scaling, if available
    try:
        # Drill down to calibrated -> base pipeline -> logreg
        base = model.base_estimator
        logreg = base.named_steps["logreg"]
        scaler = base.named_steps["scaler"]
        coefs = np.abs(logreg.coef_[0])
        names = df.drop(columns=["Class"]).columns
        imp = pd.DataFrame({"feature": names, "importance": coefs})
        imp.sort_values("importance", ascending=False, inplace=True)
        return imp.head(n)
    except Exception:
        names = df.drop(columns=["Class"]).columns
        return pd.DataFrame({"feature": names[:n], "importance": np.nan})


# ---- Gradio UI ----

def ui_train(file, test_size, budget, epsilon, seed):
    try:
        df = load_csv(file)
        res = train_model(df, test_size=test_size, random_state=seed)
        pr_png = plot_pr_curve(res.pr_curve, res.ap)

        # default budget if None
        if budget <= 0:
            budget = max(1, int(len(df) * 0.02))
        sel = select_candidates(df, res.model, budget=budget, epsilon=epsilon, seed=seed)

        # importance
        imp = quick_importance(df, res.model, n=12)

        # Render tables as TSV strings (Gradio DataFrames sometimes truncate)
        sel_show = sel.head(min(50, len(sel))).to_csv(sep='\t', index=False)
        imp_show = imp.to_csv(sep='\t', index=False)

        summary = (
            f"AP (PR-AUC): {res.ap:.4f}\n"
            f"Threshold@~2% Top-K: {res.threshold_at_k:.6f}\n"
            f"Selected (budget): {len(sel)} rows | epsilon={epsilon:.2f} (random share)\n"
            f"Top columns: {', '.join(res.feature_names[:6])}..."
        )

        return summary, pr_png, sel_show, imp_show
    except Exception as e:
        return f"Error: {e}", None, None, None


def build_app():
    with gr.Blocks(title="Fraud-Selector v0.1") as demo:
        gr.Markdown("""
        # Fraud-Selector v0.1
        Minimaler Prototyp: trainiert ein kalibriertes LogReg-Modell auf dem Credit-Card-Fraud-Datensatz
        und wählt eine **Arbeitsliste** aus mit *Budget* und *Zufallsanteil (epsilon)*.
        
        **Hinweis:** V1..V28 sind PCA-Komponenten. Erklärungen zeigen Feature-Importanzen innerhalb dieser Komponenten.
        Später werden hier eure Rechnungs-Features angezeigt.
        """)

        with gr.Row():
            file = gr.File(label="CSV hochladen (creditcard.csv)")
        with gr.Row():
            test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="Testanteil")
            budget = gr.Number(value=500, precision=0, label="Prüf-Budget (Zeilen)")
            epsilon = gr.Slider(0.0, 0.5, value=0.10, step=0.01, label="Zufallsanteil ε")
            seed = gr.Number(value=42, precision=0, label="Seed")

        run_btn = gr.Button("Trainieren & Auswahl erzeugen")

        summary = gr.Textbox(label="Zusammenfassung", lines=5)
        pr_plot = gr.Image(label="PR-Kurve", type="numpy")
        sel_tbl = gr.Textbox(label="Auswahl (Top 50, TSV)", lines=20)
        imp_tbl = gr.Textbox(label="Feature-Importanz (Top 12, TSV)", lines=12)

        run_btn.click(ui_train, inputs=[file, test_size, budget, epsilon, seed], outputs=[summary, pr_plot, sel_tbl, imp_tbl])

        gr.Markdown("""
        ## Erklärung
        - **Budget**: Anzahl der täglich zu prüfenden Zeilen.
        - **ε (epsilon)**: Zufallsanteil der Auswahl (Exploration). 0.1 = 10% Random.
        - **AP (PR-AUC)**: geeignete Metrik bei starken Klassenungleichgewichten.
        - **Threshold@~2%**: Score-Schwelle der Top-2% im Testsplit (Heuristik für Startbudget).
        
        > Nächste Iteration: Off-Policy-Evaluation (IPS/DR), SHAP-Visualisierungen, Schema-Adapter für Rechnungsdaten.
        """)
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()

