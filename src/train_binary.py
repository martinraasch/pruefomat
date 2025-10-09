"""Training script for binary fraud verification triage."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from eval_utils import (compute_cost_simulation, plot_lift_chart,
                        plot_precision_at_k, plot_precision_recall, pr_curve,
                        precision_recall_at_k)

sns.set_theme(style="whitegrid")


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

GERMAN_STOPWORDS = {
    "aber",
    "alle",
    "allem",
    "allen",
    "aller",
    "alles",
    "als",
    "also",
    "am",
    "an",
    "ander",
    "andere",
    "anderen",
    "anderer",
    "anderes",
    "andern",
    "andernfalls",
    "auch",
    "auf",
    "aus",
    "bei",
    "beim",
    "bin",
    "bis",
    "bist",
    "da",
    "dadurch",
    "daher",
    "damit",
    "dann",
    "der",
    "den",
    "des",
    "dem",
    "die",
    "das",
    "dass",
    "dazu",
    "dein",
    "deine",
    "demnach",
    "denn",
    "derer",
    "deren",
    "dessen",
    "doch",
    "dort",
    "du",
    "durch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "er",
    "es",
    "etwa",
    "euch",
    "für",
    "gegen",
    "hab",
    "haben",
    "hat",
    "hatte",
    "hatten",
    "hier",
    "hin",
    "hinter",
    "ich",
    "ihm",
    "ihn",
    "ihr",
    "ihre",
    "ihrem",
    "ihren",
    "ihrer",
    "ihres",
    "im",
    "in",
    "ins",
    "ist",
    "ja",
    "jede",
    "jedem",
    "jeden",
    "jeder",
    "jedes",
    "jener",
    "jenes",
    "jetzt",
    "kann",
    "kein",
    "keine",
    "keinem",
    "keinen",
    "keiner",
    "keines",
    "können",
    "könnte",
    "machen",
    "man",
    "manche",
    "manchem",
    "manchen",
    "mancher",
    "manches",
    "mehr",
    "mein",
    "meine",
    "mit",
    "nach",
    "nicht",
    "nichts",
    "noch",
    "nun",
    "nur",
    "ob",
    "oder",
    "ohne",
    "sehr",
    "sein",
    "seine",
    "seinem",
    "seinen",
    "seiner",
    "seines",
    "selbst",
    "sich",
    "sie",
    "sind",
    "so",
    "solche",
    "solchem",
    "solchen",
    "solcher",
    "solches",
    "soll",
    "sollen",
    "sollte",
    "sondern",
    "sonst",
    "über",
    "um",
    "und",
    "uns",
    "unser",
    "unsere",
    "unter",
    "viel",
    "vom",
    "von",
    "vor",
    "wann",
    "warum",
    "was",
    "weg",
    "weil",
    "weiter",
    "welche",
    "welchem",
    "welchen",
    "welcher",
    "welches",
    "wenn",
    "wer",
    "werde",
    "werden",
    "wie",
    "wieder",
    "wird",
    "wäre",
    "wo",
    "wollen",
    "wollte",
    "würde",
    "würden",
    "zu",
    "zum",
    "zur",
    "zwischen",
}


@dataclass
class Schema:
    columns: Dict[str, Dict[str, str]]
    options: Dict[str, object]
    label_name: str
    label_positive_rule: str


class TopNCategories(BaseEstimator, TransformerMixin):
    def __init__(self, top_n: int = 300, other_label: str = "other"):
        self.top_n = top_n
        self.other_label = other_label

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.columns_ = df.columns.tolist()
        self.categories_ = {}
        for col in self.columns_:
            vc = df[col].astype(str).value_counts()
            self.categories_[col] = vc.head(self.top_n).index.tolist()
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.columns_)
        for col in self.columns_:
            top = set(self.categories_.get(col, []))
            df[col] = df[col].astype(str).where(df[col].astype(str).isin(top), self.other_label)
        return df.values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_)


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        series = pd.Series(X.ravel() if hasattr(X, "ravel") else X)
        return series.fillna("").apply(_normalize_text).values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(["notes_text"])
        return np.asarray(input_features)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = _replace_patterns(text)
    tokens = [tok for tok in text.split() if tok not in GERMAN_STOPWORDS]
    return " ".join(tokens)


def _replace_patterns(text: str) -> str:
    text = re.sub(r"[0-9]{2,}", " __NUM__ ", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", " __MAIL__ ", text)
    text = re.sub(r"\+?[0-9][0-9\s/-]{6,}", " __PHONE__ ", text)
    text = re.sub(r"[^a-zäöüß\s_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary fraud pipeline training")
    parser.add_argument("--excel", required=True, help="Excel file path")
    parser.add_argument("--sheet", default="0", help="Sheet index or name")
    parser.add_argument("--config", required=True, help="Schema YAML")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--topk", default="0.05,0.10,0.20", help="Comma separated top-k fractions")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--topn_categories", type=int, default=300)
    parser.add_argument("--min_category_frequency", type=float, default=0.01)
    parser.add_argument("--kosten_pruefung", type=float, default=50.0)
    parser.add_argument("--kosten_fehler", type=float, default=500.0)
    parser.add_argument("--budget_anteil", type=float, default=0.1)
    parser.add_argument("--predictions-out", dest="predictions_out", default=None, help="CSV output for validation predictions")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_schema(path: str) -> Schema:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Schema(
        columns=data.get("columns", {}),
        options=data.get("options", {}),
        label_name=data.get("label", {}).get("name"),
        label_positive_rule=data.get("label", {}).get("positive_if", ">=2"),
    )


def parse_money(series: pd.Series, symbols: Iterable[str]) -> pd.Series:
    pattern = r"[^0-9,.-]"
    repl = series.astype(str)
    for sym in symbols:
        repl = repl.str.replace(sym, "", regex=False)
    repl = repl.str.replace(pattern, "", regex=True)
    repl = repl.str.replace(r"\.(?=\d{3}(\D|$))", "", regex=True)
    repl = repl.str.replace(",", ".", regex=False)
    return pd.to_numeric(repl, errors="coerce")


def parse_date(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    series = series.replace({"": pd.NA, "-": pd.NA})
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    if parsed.notna().sum() < len(parsed) * 0.6:
        parsed_alt = pd.to_datetime(series, errors="coerce", dayfirst=not dayfirst)
        parsed = parsed.combine_first(parsed_alt)
    return parsed


def parse_flag(series: pd.Series) -> pd.Series:
    truthy = {"x", "ja", "true", "1", "yes"}
    series = series.fillna("").astype(str).str.lower().str.strip()
    return series.isin(truthy)


def apply_schema(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    df = df.copy()
    options = schema.options or {}
    dayfirst = bool(options.get("dayfirst", True))
    symbols = options.get("currency_symbols", ["€", "EUR", " "])
    for col, meta in schema.columns.items():
        if col not in df.columns:
            continue
        ctype = meta.get("type")
        if ctype == "money":
            df[f"{col}_parsed"] = parse_money(df[col], symbols)
        elif ctype == "date":
            df[col] = parse_date(df[col], dayfirst=dayfirst)
        elif ctype == "flag":
            df[col] = parse_flag(df[col])
        elif ctype == "text":
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = df[col].fillna("")
    return df


def engineer_features(df: pd.DataFrame, schema: Schema) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    label_col = schema.label_name
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col} missing")
    label_series = df[label_col]
    label_numeric = pd.to_numeric(label_series, errors="coerce").fillna(0)
    positive_mask = label_numeric >= 2
    y = positive_mask.astype(int)

    flag_cols = [col for col, meta in schema.columns.items() if meta.get("type") == "flag"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    amount_col = "Betrag_parsed" if "Betrag_parsed" in df.columns else None
    if amount_col is None and "Betrag" in df.columns:
        df["Betrag_parsed"] = parse_money(df["Betrag"], schema.options.get("currency_symbols", []))
        amount_col = "Betrag_parsed"
    df["betrag_log"] = np.log1p(df[amount_col].clip(lower=0)) if amount_col else 0.0

    if "Belegdatum" in df.columns and "Fällig" in df.columns:
        df["tage_bis_faellig"] = (df["Fällig"] - df["Belegdatum"]).dt.days
    else:
        df["tage_bis_faellig"] = np.nan
    df["ist_ueberfaellig"] = (df["tage_bis_faellig"].fillna(0) < 0).astype(int)

    text_cols = [col for col, meta in schema.columns.items() if meta.get("type") == "text"]
    notes = (
        df[text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    df["notes_text"] = notes

    drop_cols = [schema.label_name]
    feature_df = df.drop(columns=drop_cols, errors="ignore")
    return feature_df, y


def build_preprocessor(schema: Schema, top_n: int, min_frequency: float) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    numeric_cols = ["Betrag_parsed", "betrag_log", "tage_bis_faellig", "ist_ueberfaellig"]
    flag_cols = [col for col, meta in schema.columns.items() if meta.get("type") == "flag"]
    numeric_cols.extend(flag_cols)
    numeric_cols = list(dict.fromkeys(numeric_cols))
    cat_cols = [col for col, meta in schema.columns.items() if meta.get("type") == "categorical"]
    cat_cols = list(dict.fromkeys(cat_cols))
    cat_cols = [col for col in cat_cols if col in schema.columns]
    text_cols = ["notes_text"]

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("topn", TopNCategories(top_n=top_n)),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_frequency,
                ),
            ),
        ]
    )

    text_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="")),
            ("normalize", TextNormalizer()),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=20000,
                    stop_words=list(GERMAN_STOPWORDS),
                ),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, cat_cols),
            ("text", text_pipeline, text_cols),
        ],
        remainder="drop",
    )
    return transformer, numeric_cols, cat_cols, text_cols


def prepare_classifiers(random_state: int) -> Dict[str, BaseEstimator]:
    classifiers: Dict[str, BaseEstimator] = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "rf": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        classifiers["xgb"] = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=random_state,
        )
    except Exception:
        pass
    return classifiers


@dataclass
class FoldResult:
    metrics: Dict[str, float]
    scores: np.ndarray
    y_true: np.ndarray


def evaluate_fold(
    preprocessor: ColumnTransformer,
    classifiers: Dict[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    top_ks: List[float],
    cost_review: float,
    cost_miss: float,
    budget_fraction: float,
) -> Tuple[str, FoldResult, Dict[str, Dict[str, float]]]:
    candidate_metrics: Dict[str, Dict[str, float]] = {}
    best_name = None
    best_pr_auc = -np.inf
    for name, clf in classifiers.items():
        pipeline = Pipeline([("prep", clone(preprocessor)), ("clf", clone(clf))])
        pipeline.fit(X_train, y_train)
        val_scores = pipeline.predict_proba(X_val)[:, -1]
        pr_auc = average_precision_score(y_val, val_scores)
        roc_auc = roc_auc_score(y_val, val_scores)
        preds = (val_scores >= 0.5).astype(int)
        metrics_dict = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "recall": recall_score(y_val, preds, zero_division=0),
            "precision": precision_score(y_val, preds, zero_division=0),
            "f1": f1_score(y_val, preds, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_val, preds),
        }
        candidate_metrics[name] = metrics_dict
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name
    assert best_name is not None

    best_clf = clone(classifiers[best_name])
    calibrator = CalibratedClassifierCV(best_clf, method="isotonic", cv=3, n_jobs=-1)
    pipeline = Pipeline([("prep", clone(preprocessor)), ("calib", calibrator)])
    pipeline.fit(X_train, y_train)
    val_scores = pipeline.predict_proba(X_val)[:, -1]
    val_preds = (val_scores >= 0.5).astype(int)

    fold_metrics = {
        "roc_auc": roc_auc_score(y_val, val_scores),
        "pr_auc": average_precision_score(y_val, val_scores),
        "recall": recall_score(y_val, val_preds, zero_division=0),
        "precision": precision_score(y_val, val_preds, zero_division=0),
        "f1": f1_score(y_val, val_preds, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_val, val_preds),
    }
    base_rate = float(np.mean(y_val))
    for k in top_ks:
        prec_k, rec_k = precision_recall_at_k(y_val, val_scores, k)
        fold_metrics[f"precision@{int(k*100)}"] = prec_k
        fold_metrics[f"recall@{int(k*100)}"] = rec_k
        fold_metrics[f"lift@{int(k*100)}"] = (prec_k / base_rate) if base_rate > 0 else float("nan")
        fold_metrics[f"cost@{int(k*100)}"] = compute_cost_simulation(
            y_val, val_scores, k, cost_review, cost_miss
        )
    fold_metrics["cost@budget"] = compute_cost_simulation(
        y_val, val_scores, budget_fraction, cost_review, cost_miss
    )

    return best_name, FoldResult(metrics=fold_metrics, scores=val_scores, y_true=y_val), candidate_metrics


def aggregate_metrics(fold_results: List[FoldResult]) -> Dict[str, Dict[str, float]]:
    keys = fold_results[0].metrics.keys()
    agg: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.array([fr.metrics[key] for fr in fold_results])
        agg[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return agg


def plot_feature_importance(features: np.ndarray, feature_names: np.ndarray, path: str, top_k: int = 30) -> None:
    order = np.argsort(features)[::-1][:top_k]
    plt.figure(figsize=(8, max(4, top_k * 0.25)))
    plt.barh(feature_names[order][::-1], features[order][::-1])
    plt.xlabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@dataclass
class ArtifactPaths:
    metrics: Path
    pr_curve: Path
    precision_k: Path
    lift_k: Path
    feature_importance: Path
    shap_summary: Path
    shap_examples: Path


class PreprocessorModelWrapper:
    def __init__(self, preprocessor: ColumnTransformer, calibrator: CalibratedClassifierCV):
        self.preprocessor = preprocessor
        self.calibrator = calibrator

    def predict_proba(self, X):
        Xt = self.preprocessor.transform(X)
        return self.calibrator.predict_proba(Xt)

    def predict(self, X):
        return self.predict_proba(X)[:, -1] >= 0.5


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    schema = load_schema(args.config)
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df_raw = pd.read_excel(args.excel, sheet_name=sheet, dtype=str)
    df_parsed = apply_schema(df_raw, schema)
    features_df, y = engineer_features(df_parsed, schema)

    preprocessor, numeric_cols, cat_cols, text_cols = build_preprocessor(
        schema, args.topn_categories, args.min_category_frequency
    )

    classifiers = prepare_classifiers(args.random_state)
    k_values = [float(x) for x in args.topk.split(",") if x]
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.random_state)

    fold_results: List[FoldResult] = []
    candidate_overview: Dict[str, List[float]] = {name: [] for name in classifiers}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features_df, y), start=1):
        X_train = features_df.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = features_df.iloc[val_idx]
        y_val = y.iloc[val_idx]
        preproc_clone = clone(preprocessor)
        best_name, fold_result, candidate_metrics = evaluate_fold(
            preproc_clone,
            classifiers,
            X_train,
            y_train,
            X_val,
            y_val,
            k_values,
            args.kosten_pruefung,
            args.kosten_fehler,
            args.budget_anteil,
        )
        fold_results.append(fold_result)
        for name, metrics_dict in candidate_metrics.items():
            candidate_overview[name].append(metrics_dict["pr_auc"])
        print(f"Fold {fold_idx}: best={best_name} PR-AUC={fold_result.metrics['pr_auc']:.3f}")

    aggregated = aggregate_metrics(fold_results)

    best_global = max(candidate_overview.items(), key=lambda item: np.mean(item[1]))[0]
    base_estimator = clone(classifiers[best_global])
    preprocessor_final = clone(preprocessor)
    Xt_full = preprocessor_final.fit_transform(features_df, y)
    calibrator_final = CalibratedClassifierCV(clone(classifiers[best_global]), method="isotonic", cv=3, n_jobs=-1)
    calibrator_final.fit(Xt_full, y)
    fitted_estimator = clone(classifiers[best_global])
    fitted_estimator.fit(Xt_full, y)

    wrapper = PreprocessorModelWrapper(preprocessor_final, calibrator_final)
    joblib.dump(preprocessor_final, outdir / "preprocessor.joblib")
    joblib.dump(wrapper, outdir / "model.joblib")

    feature_names = np.array(preprocessor_final.get_feature_names_out())

    full_scores = wrapper.predict_proba(features_df)[:, -1]
    full_preds = (full_scores >= 0.5).astype(int)
    predictions_table = pd.DataFrame(
        {
            "fraud_score": full_scores * 100.0,
            "prediction": full_preds,
            "actual": y,
        }
    )
    predictions_table.sort_values("fraud_score", ascending=False, inplace=True)
    predictions_out = Path(args.predictions_out) if args.predictions_out else (outdir / "predictions.csv")
    ensure_directory(predictions_out)
    predictions_table.to_csv(predictions_out, index=False)

    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "kfolds": args.kfolds,
                "topk": k_values,
                "metrics": aggregated,
                "best_model": best_global,
            },
            fh,
            indent=2,
        )

    all_scores = np.concatenate([fr.scores for fr in fold_results])
    all_true = np.concatenate([fr.y_true for fr in fold_results])
    pr = pr_curve(all_true, all_scores)
    plot_precision_recall(pr, aggregated["pr_auc"]["mean"], str(outdir / "pr_curve.png"))

    precisions = []
    lifts = []
    base_rate = float(np.mean(all_true))
    for k in k_values:
        prec, _ = precision_recall_at_k(all_true, all_scores, k)
        precisions.append(prec)
        lifts.append(prec / base_rate if base_rate > 0 else float("nan"))
    plot_precision_at_k(k_values, precisions, str(outdir / "precision_at_k.png"))
    plot_lift_chart(k_values, lifts, str(outdir / "lift.png"))

    if hasattr(fitted_estimator, "feature_importances_"):
        importances = fitted_estimator.feature_importances_
    elif hasattr(fitted_estimator, "coef_"):
        importances = np.abs(fitted_estimator.coef_).ravel()
    else:
        importances = None
    if importances is not None:
        plot_feature_importance(importances, feature_names, str(outdir / "feature_importance.png"))

    try:
        Xt_dense = Xt_full.toarray() if hasattr(Xt_full, "toarray") else Xt_full
        if hasattr(fitted_estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(fitted_estimator)
            shap_values = explainer.shap_values(Xt_dense)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
        else:
            explainer = shap.LinearExplainer(fitted_estimator, Xt_dense)
            shap_values = explainer(Xt_dense).values
        shap.summary_plot(shap_values, Xt_dense, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_summary.png", dpi=160)
        plt.close()

        top_idx = np.argsort(full_scores)[::-1][:5]
        shap_values_top = shap_values[top_idx]
        shap.summary_plot(shap_values_top, Xt_dense[top_idx], feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_examples.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"SHAP computation failed: {exc}")

    feature_map = {
        "numeric": numeric_cols,
        "categorical": cat_cols,
        "text": text_cols,
        "feature_names": feature_names.tolist(),
    }
    with open(outdir / "feature_map.json", "w", encoding="utf-8") as fh:
        json.dump(feature_map, fh, indent=2)

    with open(outdir / "schema_used.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "schema": schema.columns,
                "options": schema.options,
                "label": schema.label_name,
                "positive_rule": schema.label_positive_rule,
            },
            fh,
            indent=2,
        )

    print(json.dumps({
        "metrics": str(metrics_path),
        "model": str(outdir / "model.joblib"),
        "preprocessor": str(outdir / "preprocessor.joblib"),
        "predictions": str(predictions_out),
    }))


if __name__ == "__main__":
    main()
