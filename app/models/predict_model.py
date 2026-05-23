import joblib
import numpy as np
import os

MODEL_DIR = "trained_models"
_cache = {}


def _load_artifact(disease: str) -> dict:
    if disease not in _cache:
        path = os.path.join(MODEL_DIR, f"{disease}_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found at {path}. "
                "Please run: python -m app.models.train_all_models"
            )
        _cache[disease] = joblib.load(path)
    return _cache[disease]


def predict_disease(disease: str, input_values: dict) -> dict:
    """
    Predict disease risk for any of the 4 diseases.
    
    Args:
        disease: 'diabetes', 'heart', 'kidney', 'liver'
        input_values: dict of feature_name → value
    
    Returns:
        prediction (0/1), risk_score (0–1), risk_level (Low/Medium/High)
    """
    artifact = _load_artifact(disease)
    model = artifact["model"]
    imputer = artifact["imputer"]
    feature_names = artifact["feature_names"]

    import pandas as pd
    # Build DataFrame in correct order to avoid feature names warning
    input_df = pd.DataFrame([[input_values.get(f, 0) for f in feature_names]], columns=feature_names)

    # Apply same imputer used during training
    input_imputed = imputer.transform(input_df)

    prediction = int(model.predict(input_imputed)[0])
    probability = float(model.predict_proba(input_imputed)[0][1])

    risk_level = "Low"
    if probability > 0.7:
        risk_level = "High"
    elif probability > 0.4:
        risk_level = "Medium"

    return {
        "prediction": prediction,
        "risk_score": round(probability, 4),
        "risk_level": risk_level,
        "confidence_percent": round(probability * 100, 1),
        "disease": disease,
    }