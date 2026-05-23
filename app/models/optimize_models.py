# app/models/optimize_models.py

import os
import warnings
import joblib
import optuna
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

DATASET_DIR = "datasets"
MODEL_DIR = "trained_models"


# =========================================================
# LOAD + PREPARE DIABETES DATASET
# =========================================================

def load_diabetes():

    df = pd.read_csv(os.path.join(DATASET_DIR, "diabetes.csv"))

    zero_not_allowed = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    df[zero_not_allowed] = df[zero_not_allowed].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    imputer = SimpleImputer(strategy="median")

    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    sm = SMOTE(random_state=42)

    X_train_res, y_train_res = sm.fit_resample(
        X_train,
        y_train
    )

    return (
        X_train_res,
        X_test,
        y_train_res,
        y_test,
        imputer,
        list(X.columns)
    )


# =========================================================
# OPTUNA OBJECTIVE
# =========================================================

def objective(trial, X_train, y_train):

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            600
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0,
            5
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            0,
            5
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0,
            5
        ),

        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    model = XGBClassifier(**params)

    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    ).mean()

    return score


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\nLoading dataset...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        imputer,
        feature_names
    ) = load_diabetes()

    print("Dataset loaded.")
    print("Starting Optuna optimization...\n")

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train
        ),
        n_trials=50
    )

    print("\n==============================")
    print("BEST PARAMETERS")
    print("==============================")

    print(study.best_params)

    print("\nBest ROC-AUC:")
    print(study.best_value)

    # ==========================================
    # TRAIN FINAL MODEL WITH BEST PARAMS
    # ==========================================

    best_model = XGBClassifier(
        **study.best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    best_model.fit(X_train, y_train)

    artifact = {
        "model": best_model,
        "imputer": imputer,
        "feature_names": feature_names,
    }

    save_path = os.path.join(
        MODEL_DIR,
        "diabetes_model_optimized.pkl"
    )

    joblib.dump(artifact, save_path)

    print(f"\nOptimized model saved at: {save_path}")