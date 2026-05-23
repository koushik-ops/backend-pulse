# app/explainability/shap_explainer.py

import shap
import numpy as np
import joblib
import os

MODEL_DIR = "trained_models"

# Cache loaded artifacts to avoid re-loading on every request
_cache = {}


def _load_artifact(disease: str) -> dict:

    if disease not in _cache:

        path = os.path.join(
            MODEL_DIR,
            f"{disease}_model.pkl"
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Model not found: {path}. "
                f"Run train_all_models.py first."
            )

        _cache[disease] = joblib.load(path)

    return _cache[disease]


def explain_prediction(
    disease: str,
    input_values: dict
) -> dict:

    """
    Returns true per-instance SHAP values.

    Args:
        disease:
            'diabetes', 'heart', 'kidney', 'liver'

        input_values:
            {feature_name: value}

    Returns:
        {
            shap_values,
            base_value,
            top_features
        }
    """

    artifact = _load_artifact(disease)

    model = artifact["model"]
    imputer = artifact["imputer"]
    feature_names = artifact["feature_names"]

    import pandas as pd
    # ==========================================
    # BUILD INPUT DATAFRAME IN CORRECT ORDER
    # ==========================================

    input_df = pd.DataFrame([
        [
            input_values.get(f, 0)
            for f in feature_names
        ]
    ], columns=feature_names)

    # ==========================================
    # APPLY SAME IMPUTATION AS TRAINING
    # ==========================================

    input_imputed = imputer.transform(input_df)

    # ==========================================
    # CREATE SHAP EXPLAINER
    # ==========================================

    explainer = shap.TreeExplainer(
    model,
    feature_perturbation="tree_path_dependent"
    )
    explainer.model.original_model.set_param(
    {"base_score": 0.5}
    )

    shap_values = explainer.shap_values(
        input_imputed
    )

    # ==========================================
    # HANDLE BINARY CLASSIFICATION FORMAT
    # ==========================================

    if isinstance(shap_values, list):

        sv = shap_values[1][0]

    else:

        sv = shap_values[0]

    # ==========================================
    # BASE VALUE
    # ==========================================

    base_value = float(explainer.expected_value)

    if isinstance(
        explainer.expected_value,
        (list, np.ndarray)
    ):

        base_value = float(
            explainer.expected_value[1]
        )

    # ==========================================
    # BUILD EXPLANATIONS
    # ==========================================

    explanations = {}

    for i, feature in enumerate(feature_names):

        shap_val = float(sv[i])

        explanations[feature] = {

            "value": float(
                input_imputed[0][i]
            ),

            "shap": round(
                shap_val,
                5
            ),

            "direction":
                "increases_risk"
                if shap_val > 0
                else "decreases_risk"
        }

    # ==========================================
    # SORT TOP FEATURES
    # ==========================================

    sorted_features = sorted(
        explanations.items(),
        key=lambda x: abs(x[1]["shap"]),
        reverse=True
    )

    top_features = [

        {
            "feature": k,
            "value": v["value"],
            "shap": v["shap"],
            "direction": v["direction"],
        }

        for k, v in sorted_features[:5]

    ]

    return {

        "shap_values": explanations,

        "base_value": round(
            base_value,
            5
        ),

        "top_features": top_features,
    }