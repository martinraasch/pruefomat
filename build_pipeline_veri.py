#!/usr/bin/env python3
# build_pipeline_veri.py
# Usage-Beispiele:
#   Nur Preprocessing (ohne Ampel):
#     python build_pipeline_veri.py --excel Veri-Bsp.xlsx --sheet 0
#   Mit Training auf Zielspalte "Ampel":
#     python build_pipeline_veri.py --excel Veri-Bsp.xlsx --sheet 0 --target Ampel
#   Mit lokalen Text-Embeddings statt TF-IDF:
#     python build_pipeline_veri.py --excel Veri-Bsp.xlsx --sheet 0 --target Ampel --use-embeddings

import argparse, re, sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import joblib

# ----------------------
# Hilfsfunktionen
# ----------------------

GERMAN_MONEY_PATTERN = re.compile(r"[.\s]")

def to_numeric_series(s: pd.Series) -> pd.Series:
    """Konvertiert robust Beträge: ' 43,914.51 ' oder '43.914,51' -> float."""
    ss = s.astype(str).str.strip()
    # de: Komma als Dezimaltrenner
    de_mask = ss.str.contains(r",\d{1,2}$")
    ss_de = ss[de_mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    # en/sonst: entferne Tausendertrennzeichen/Leerzeichen
    ss_en = ss[~de_mask].str.replace(GERMAN_MONEY_PATTERN, "", regex=True)
    merged = pd.concat([ss_de, ss_en]).reindex(s.index)
    return pd.to_numeric(merged, errors="coerce")

def to_datetime_series(s: pd.Series) -> pd.Series:
    """Konvertiert robust Datumsangaben inkl. '-', 'diverse' -> NaT."""
    x = s.astype(str).str.strip().replace({"-": np.nan, "diverse": np.nan, "": np.nan})
    # erst dayfirst=True versuchen, dann False
    for dayfirst in (True, False):
        dt = pd.to_datetime(x, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)
        if dt.notna().mean() > 0.6:
            return dt
    return pd.to_datetime(x, errors="coerce", dayfirst=True)

def normalize_colname(c):
    c = str(c).strip().replace("\n", " ")
    c = c.replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss")
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^A-Za-z0-9_]", "", c)
    return c

# ----------------------
# Custom Transformer
# ----------------------

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Erzeugt Datumsfeatures und tage_bis_faellig aus Belegdatum/Faellig."""
    def __init__(self, beleg_col: Optional[str], faellig_col: Optional[str], date_cols: List[str]):
        self.beleg_col = beleg_col
        self.faellig_col = faellig_col
        self.date_cols = date_cols or []

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.out_cols_ = []
        for c in self.date_cols:
            if c in df.columns:
                self.out_cols_ += [f"{c}_year", f"{c}_month", f"{c}_weekday", f"{c}_is_month_end"]
        if self.beleg_col and self.faellig_col:
            self.out_cols_ += ["tage_bis_faellig"]
        return self

    def transform(self, X):
        check_is_fitted(self, "out_cols_")
        df = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=df.index)

        for c in self.date_cols:
            if c not in df.columns: 
                continue
            dt = to_datetime_series(df[c])
            out[f"{c}_year"] = dt.dt.year
            out[f"{c}_month"] = dt.dt.month
            out[f"{c}_weekday"] = dt.dt.weekday
            out[f"{c}_is_month_end"] = dt.dt.is_month_end.astype(float)

        if self.beleg_col and self.faellig_col:
            due = to_datetime_series(df[self.faellig_col])
            iss = to_datetime_series(df[self.beleg_col])
            out["tage_bis_faellig"] = (due - iss).dt.days

        return out

class MergeDateFeatures(BaseEstimator, TransformerMixin):
    """Hängt die von DateFeatureExtractor erzeugten Spalten an das Original-DataFrame an."""
    def __init__(self, datefe: DateFeatureExtractor):
        self.datefe = datefe

    def fit(self, X, y=None):
        self.datefe.fit(X, y)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        d = self.datefe.transform(X)
        for c in d.columns:
            X[c] = d[c]
        return X

class TextConcatenator(BaseEstimator, TransformerMixin):
    """Fasst mehrere Textspalten zu einer einzelnen Serie zusammen (für TF-IDF/Embeddings)."""
    def __init__(self, text_cols: List[str], out_col="TEXT_ALL"):
        self.text_cols = text_cols or []
        self.out_col = out_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        if not self.text_cols:
            return pd.DataFrame({self.out_col: [""] * len(df)})
        conc = df[self.text_cols].astype(str).replace("nan","").apply(
            lambda r: " ".join([x for x in r if x and x != "nan"]), axis=1
        )
        return pd.DataFrame({self.out_col: conc})

class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):
    """Optionaler lokaler Embedding-Encoder (CPU-tauglich)."""
    def __init__(self, text_col="TEXT_ALL", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.text_col = text_col
        self.model_name = model_name
        self.model_ = None

    def fit(self, X, y=None):
        texts = pd.Series(X[self.text_col]).fillna("").astype(str).tolist()
        # Lazy import (install: sentence-transformers)
        from sentence_transformers import SentenceTransformer
        self.model_ = SentenceTransformer(self.model_name)
        # einmal durchlaufen, sorgt für Download/Caching & Shapes
        _ = self.model_.encode(texts[:8])
        return self

    def transform(self, X):
        check_is_fitted(self, "model_")
        texts = pd.Series(X[self.text_col]).fillna("").astype(str).tolist()
        vecs = self.model_.encode(texts, normalize_embeddings=True)
        return vecs  # 2D numpy array

# ----------------------
# Hauptlogik
# ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Pfad zur Excel-Datei")
    ap.add_argument("--sheet", default="0", help="Sheet-Name oder Index")
    ap.add_argument("--target", default=None, help="Zielspalte (z. B. Ampel)")
    ap.add_argument("--use-embeddings", action="store_true", help="Lokale Embeddings statt TF-IDF für Text")
    ap.add_argument("--tfidf-maxfeatures", type=int, default=5000)
    ap.add_argument("--model-out", default="model.joblib")
    ap.add_argument("--prep-out", default="preprocessor.joblib")
    ap.add_argument("--report-out", default="report.txt")
    args = ap.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet

    # Excel laden (alles als str, damit wir formatrobust parsen können)
    df_raw = pd.read_excel(args.excel, sheet_name=sheet, dtype=str)
    original_cols = list(df_raw.columns)

    # Spaltennamen normalisieren (für stabile Referenzen)
    df_raw.columns = [normalize_colname(c) for c in df_raw.columns]

    # Mapping der bekannten Spalten (aus deiner Liste)
    # Original -> Normalisiert (zum Verständnis)
    # Land -> Land
    # BUK -> BUK
    # DEB Name -> DEB_Name
    # Rechnungsnummer -> Rechnungsnummer
    # Belegdatum -> Belegdatum
    # Betrag -> Betrag
    # Fällig -> Faellig
    # Debitor -> Debitor
    # Ampel -> Ampel
    # Maßnahme 2025 -> Massnahme_2025
    # MA DRM -> MA_DRM
    # Datum -> Datum
    # Hinweise -> Hinweise
    # Rücklauf -> Ruecklauf
    # WV -> WV
    # Ticket-nummer -> Ticketnummer
    # Rückmeldung erhalten -> Rueckmeldung_erhalten
    # negativ -> negativ

    # Zielspalte prüfen
    target = args.target
    if target:
        target = normalize_colname(target)
        if target not in df_raw.columns:
            print(f"⚠️ Zielspalte '{args.target}' nicht gefunden (nach Normalisierung '{target}'). Es wird kein Modell trainiert.")
            target = None

    # Typ-Set/Liste für deine Excel (fester Plan statt kompletter Auto-Erkennung)
    amount_col = "Betrag" if "Betrag" in df_raw.columns else None
    beleg_col  = "Belegdatum" if "Belegdatum" in df_raw.columns else None
    faellig_col= "Faellig" if "Faellig" in df_raw.columns else None

    # Categorical-Felder (Low-Cardinality)
    categorical_cols = [c for c in [
        "Land","BUK","DEB_Name","Rechnungsnummer","Debitor","MA_DRM","WV","Ticketnummer",
        "Rueckmeldung_erhalten","negativ"
    ] if c in df_raw.columns]

    # Numerik (explizit Betrag; später kommen Datums-Features dazu)
    numeric_cols = []
    if amount_col and amount_col in df_raw.columns:
        df_raw[amount_col] = to_numeric_series(df_raw[amount_col])
        numeric_cols.append(amount_col)
        # Bonus: Log-Betrag als Feature (robust gegen Schiefe)
        df_raw["Betrag_log1p"] = np.log1p(df_raw[amount_col].clip(lower=0))
        numeric_cols.append("Betrag_log1p")

    # Textfelder sinnvoll zusammenführen
    text_cols = [c for c in ["Hinweise","Massnahme_2025","Ruecklauf"] if c in df_raw.columns]
    # Zusätzlich kurze, semi-textuelle Felder mitnehmen (nicht doppelt, falls schon kategorial):
    short_text_candidates = [c for c in ["DEB_Name","Rechnungsnummer"] if c in df_raw.columns and c not in categorical_cols]
    text_cols += [c for c in short_text_candidates if c not in text_cols]

    # Eine Sammelspalte TEXT_ALL bauen
    txt_concat = TextConcatenator(text_cols=text_cols, out_col="TEXT_ALL")
    df_text = txt_concat.transform(df_raw)
    df_work = pd.concat([df_raw, df_text], axis=1)

    # Datumsfeatures
    date_cols = [c for c in ["Belegdatum","Faellig","Datum"] if c in df_work.columns]
    datefe = DateFeatureExtractor(beleg_col=beleg_col, faellig_col=faellig_col, date_cols=date_cols)

    # Pipeline-Parts
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    # Textpipeline: TF-IDF (Standard) oder lokale Embeddings (optional)
    if args.use_embeddings:
        text_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="")),
            # ColumnTransformer gibt 2D zurück -> greife Spalte per Funktions-Wrapper
            ("to_series", FunctionTransformer(lambda X: pd.Series(X.ravel()), validate=False)),
            ("rename", FunctionTransformer(lambda s: pd.DataFrame({"TEXT_ALL": s}), validate=False)),
            ("emb", SentenceTransformerEncoder(text_col="TEXT_ALL"))
        ])
    else:
        text_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="")),
            ("to_series", FunctionTransformer(lambda X: pd.Series(X.ravel()), validate=False)),
            ("tfidf", TfidfVectorizer(max_features=args.tfidf_maxfeatures, ngram_range=(1,2)))
        ])

    # Kleiner Import hier, damit sklearn-Versionen ohne FunctionTransformer nicht stolpern:
    from sklearn.preprocessing import FunctionTransformer

    # ColumnTransformer: wir geben Listen der existierenden Spalten rein
    used_numeric = [c for c in numeric_cols if c in df_work.columns]
    used_categorical = [c for c in categorical_cols if c in df_work.columns]
    used_text = ["TEXT_ALL"] if "TEXT_ALL" in df_work.columns else []

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, used_numeric),
            ("cat", categorical_pipe, used_categorical),
            ("text", text_pipe, used_text)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # Gesamt-Pipeline: DateFeatures zuerst mergen, dann ColumnTransformer
    full_pipeline = Pipeline([
        ("merge_dates", MergeDateFeatures(datefe)),
        ("prep", preprocessor)
    ])

    report = []
    report.append("# Build report\n")
    report.append(f"- Original columns: {original_cols}")
    report.append(f"- Normalized columns: {list(df_raw.columns)}\n")
    report.append("## Feature-Plan\n")
    report.append(f"- Betrag-Spalte: {amount_col}")
    report.append(f"- Datums-Spalten (Beleg/Faellig/Datum): {beleg_col}, {faellig_col}, {[c for c in date_cols]}")
    report.append(f"- Numerik (inkl. log1p): {used_numeric}")
    report.append(f"- Kategorisch: {used_categorical}")
    report.append(f"- Textspalten -> TEXT_ALL: {text_cols}  (Encoder: {'Embeddings' if args.use_embeddings else 'TF-IDF'})\n")

    # Fit & optional Train
    if target:
        y = df_work[target].copy()
        X = df_work.drop(columns=[target])
        # stratify nur wenn mehrere Klassen vorhanden
        strat = y if y.nunique() > 1 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)

        Ztr = full_pipeline.fit_transform(Xtr)
        Zte = full_pipeline.transform(Xte)

        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        clf.fit(Ztr, ytr)
        yhat = clf.predict(Zte)

        cr = classification_report(yte, yhat, digits=3)
        cm = confusion_matrix(yte, yhat)

        report.append("## Klassifikationsbericht (Test)\n")
        report.append("```\n" + cr + "\n```")
        report.append("Confusion Matrix (Zeilen=Ist, Spalten=Vorhersage):\n" + str(cm) + "\n")

        joblib.dump(clf, args.model_out)
        report.append(f"✔️ Modell gespeichert: {args.model_out}")
    else:
        # Nur Preprocessor fitten (z. B. für spätere Nutzung)
        _ = full_pipeline.fit_transform(df_work)
        report.append("ℹ️ Kein Ziel vorhanden — nur Preprocessor fit_transform ausgeführt.")

    joblib.dump(full_pipeline, args.prep_out)
    report.append(f"✔️ Preprocessor gespeichert: {args.prep_out}")

    Path(args.report_out).write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    sys.exit(main())