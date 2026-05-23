import os
import warnings
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

DATASET_DIR = "datasets"
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate_model(model, X_test, y_test, name):
    """Print evaluation metrics for a trained model."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"\n{'='*50}")
    print(f"  {name} Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy_score(y_test, preds):.4f}")
    print(f"  Precision : {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, preds, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, proba):.4f}")
    print(classification_report(y_test, preds, zero_division=0))


# ─────────────────────────────────────────────
# DIABETES
# ─────────────────────────────────────────────
def train_diabetes():
    print("\n>>> Training: Diabetes")
    df = pd.read_csv(os.path.join(DATASET_DIR, "diabetes.csv"))

    # Replace biologically impossible zeros with NaN for imputation
    zero_not_allowed = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_not_allowed] = df[zero_not_allowed].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Impute with median (robust to outliers)
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE to handle class imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=267,
        max_depth=8,
        learning_rate=0.03085337474327392,
        subsample=0.641335274485658,
        colsample_bytree=0.8649665314859781,
        gamma=0.01800832927275648,
        min_child_weight=1,
        reg_alpha=0.8636894384750555,
        reg_lambda=0.05977362402997091,
    )
    model.fit(X_train_res, y_train_res)
    evaluate_model(model, X_test, y_test, "Diabetes")

    # Save model + imputer + feature names together
    artifact = {
        "model": model,
        "imputer": imputer,
        "feature_names": list(X.columns),
    }
    joblib.dump(artifact, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
    print("  Saved: trained_models/diabetes_model.pkl")


# ─────────────────────────────────────────────
# HEART DISEASE
# ─────────────────────────────────────────────
def train_heart():
    print("\n>>> Training: Heart Disease")
    df = pd.read_csv(os.path.join(DATASET_DIR, "heart_disease_uci.csv"))

    # Binarize target: 0 = no disease, 1 = disease (any degree)
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["id", "num", "dataset"], errors="ignore")

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    X = df.drop("target", axis=1)
    y = df["target"]

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
       n_estimators=267,
        max_depth=8,
        learning_rate=0.03085337474327392,
        subsample=0.641335274485658,
        colsample_bytree=0.8649665314859781,
        gamma=0.01800832927275648,
        min_child_weight=1,
        reg_alpha=0.8636894384750555,
        reg_lambda=0.05977362402997091,
    )
    model.fit(X_train_res, y_train_res)
    evaluate_model(model, X_test, y_test, "Heart Disease")

    artifact = {
        "model": model,
        "imputer": imputer,
        "feature_names": list(X.columns),
        "cat_encoders": {},  # extend if you need inverse transform later
    }
    joblib.dump(artifact, os.path.join(MODEL_DIR, "heart_model.pkl"))
    print("  Saved: trained_models/heart_model.pkl")


# ─────────────────────────────────────────────
# KIDNEY DISEASE
# ─────────────────────────────────────────────
def train_kidney():
    print("\n>>> Training: Kidney Disease")
    df = pd.read_csv(os.path.join(DATASET_DIR, "kidney_disease.csv"))

    # Fix tab-corrupted label
    df["classification"] = df["classification"].str.strip()
    df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})
    df = df.dropna(subset=["classification"])

    # Drop id column
    df = df.drop(columns=["id"], errors="ignore")

    # Encode all object columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    # Map common values
    bool_maps = {
        "yes": 1, "no": 0, "present": 1, "notpresent": 0,
        "normal": 1, "abnormal": 0, "good": 1, "poor": 0
    }
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].map(bool_maps).fillna(np.nan)

    X = df.drop("classification", axis=1)
    y = df["classification"].astype(int)

    # Add keep_empty_features=True to your imputer
    imputer = SimpleImputer(strategy='mean', keep_empty_features=True) 

    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    # Kidney is already imbalanced 248:150 — apply SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=267,
        max_depth=8,
        learning_rate=0.03085337474327392,
        subsample=0.641335274485658,
        colsample_bytree=0.8649665314859781,
        gamma=0.01800832927275648,
        min_child_weight=1,
        reg_alpha=0.8636894384750555,
        reg_lambda=0.05977362402997091,
    )
    model.fit(X_train_res, y_train_res)
    evaluate_model(model, X_test, y_test, "Kidney Disease")

    artifact = {
        "model": model,
        "imputer": imputer,
        "feature_names": list(X.columns),
    }
    joblib.dump(artifact, os.path.join(MODEL_DIR, "kidney_model.pkl"))
    print("  Saved: trained_models/kidney_model.pkl")


# ─────────────────────────────────────────────
# LIVER DISEASE
# ─────────────────────────────────────────────
def train_liver():
    print("\n>>> Training: Liver Disease")
    df = pd.read_csv(
        os.path.join(DATASET_DIR, "Liver Patient Dataset (LPD)_train.csv"),
        encoding="latin-1"
    )

    # Strip non-breaking spaces from column names
    df.columns = [c.strip().replace("\xa0", "") for c in df.columns]

    # Target: 1=liver disease, 2=no disease → remap to 1=disease, 0=healthy
    df["Result"] = df["Result"].map({1: 1, 2: 0})
    df = df.dropna(subset=["Result"])

    # Encode gender
    df["Gender of the patient"] = df["Gender of the patient"].map(
        {"Male": 1, "Female": 0}
    ).fillna(0)

    X = df.drop("Result", axis=1)
    y = df["Result"].astype(int)

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=267,
        max_depth=8,
        learning_rate=0.03085337474327392,
        subsample=0.641335274485658,
        colsample_bytree=0.8649665314859781,
        gamma=0.01800832927275648,
        min_child_weight=1,
        reg_alpha=0.8636894384750555,
        reg_lambda=0.05977362402997091,
    )
    model.fit(X_train_res, y_train_res)
    evaluate_model(model, X_test, y_test, "Liver Disease")

    artifact = {
        "model": model,
        "imputer": imputer,
        "feature_names": list(X.columns),
    }
    joblib.dump(artifact, os.path.join(MODEL_DIR, "liver_model.pkl"))
    print("  Saved: trained_models/liver_model.pkl")


if __name__ == "__main__":
    train_diabetes()
    train_heart()
    train_kidney()
    train_liver()
    print("\n[SUCCESS] All models trained and saved successfully.")

