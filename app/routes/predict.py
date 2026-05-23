from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from typing import Optional
from app.models.predict_model import predict_disease
from app.explainability.shap_explainer import explain_prediction
from app.services.medical_summary import generate_enriched_summary

router = APIRouter(prefix="/predict", tags=["Prediction"])


# ── Pydantic models (one per disease) ─────────────────────────────────────────

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

    @validator("Glucose")
    def glucose_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Glucose must be > 0")
        return v

    @validator("BMI")
    def bmi_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("BMI must be > 0")
        return v


class HeartInput(BaseModel):
    age: float
    sex: int          # 0=female, 1=male
    cp: int           # chest pain type: 0-3
    trestbps: float   # resting blood pressure
    chol: float       # cholesterol
    fbs: int          # fasting blood sugar > 120 (1=true, 0=false)
    restecg: int      # resting ECG
    thalch: float     # max heart rate achieved
    exang: int        # exercise-induced angina
    oldpeak: float    # ST depression
    slope: int        # slope of peak exercise ST
    ca: Optional[float] = 0   # number of major vessels colored by fluoroscopy
    thal: Optional[float] = 2  # thalassemia: 1=normal, 2=fixed defect, 3=reversible


class KidneyInput(BaseModel):
    age: float
    bp: float     # blood pressure
    sg: float     # specific gravity
    al: float     # albumin (0-5 scale)
    su: float     # sugar (0-5 scale)
    rbc: float    # red blood cells (1=normal, 0=abnormal)
    pc: float     # pus cells (1=normal, 0=abnormal)
    pcc: float    # pus cell clumps (1=present, 0=notpresent)
    ba: float     # bacteria (1=present, 0=notpresent)
    bgr: float    # blood glucose random
    bu: float     # blood urea
    sc: float     # serum creatinine
    sod: float    # sodium
    pot: float    # potassium
    hemo: float   # hemoglobin
    pcv: float    # packed cell volume
    wc: float     # white blood cell count
    rc: float     # red blood cell count
    htn: float    # hypertension (1=yes, 0=no)
    dm: float     # diabetes mellitus (1=yes, 0=no)
    cad: float    # coronary artery disease (1=yes, 0=no)
    appet: float  # appetite (1=good, 0=poor)
    pe: float     # pedal edema (1=yes, 0=no)
    ane: float    # anemia (1=yes, 0=no)


class LiverInput(BaseModel):
    age: float                              # Age of the patient
    gender: int                             # 1=male, 0=female
    total_bilirubin: float
    direct_bilirubin: float
    alkaline_phosphotase: float
    alamine_aminotransferase: float
    aspartate_aminotransferase: float
    total_proteins: float
    albumin: float
    albumin_globulin_ratio: float


# ── Helper: map user-friendly field names to dataset column names ──────────────

LIVER_FIELD_MAP = {
    "age": "Age of the patient",
    "gender": "Gender of the patient",
    "total_bilirubin": "Total Bilirubin",
    "direct_bilirubin": "Direct Bilirubin",
    "alkaline_phosphotase": "Alkphos Alkaline Phosphotase",
    "alamine_aminotransferase": "Sgpt Alamine Aminotransferase",
    "aspartate_aminotransferase": "Sgot Aspartate Aminotransferase",
    "total_proteins": "Total Protiens",
    "albumin": "ALB Albumin",
    "albumin_globulin_ratio": "A/G Ratio Albumin and Globulin Ratio",
}


def _build_response(disease: str, input_dict: dict) -> dict:
    """Central pipeline: predict → explain → enrich → return."""
    prediction = predict_disease(disease, input_dict)
    explanation = explain_prediction(disease, input_dict)
    summary = generate_enriched_summary(
        disease=disease,
        risk_level=prediction["risk_level"],
        risk_score=prediction["risk_score"],
        top_shap_features=explanation["top_features"],
        feature_values=input_dict,
    )
    return {
        "prediction": prediction,
        "explanation": explanation,
        "summary": summary,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/diabetes")
def predict_diabetes(data: DiabetesInput):
    return _build_response("diabetes", data.dict())


@router.post("/heart")
def predict_heart(data: HeartInput):
    return _build_response("heart", data.dict())


@router.post("/kidney")
def predict_kidney(data: KidneyInput):
    return _build_response("kidney", data.dict())


@router.post("/liver")
def predict_liver(data: LiverInput):
    # Remap field names to match dataset columns
    mapped = {LIVER_FIELD_MAP.get(k, k): v for k, v in data.dict().items()}
    return _build_response("liver", mapped)


from app.federated.federated_stub import get_federated_info

@router.get("/federated-info")
def federated_info():
    """Returns federated learning metadata for dashboard."""
    return get_federated_info()

