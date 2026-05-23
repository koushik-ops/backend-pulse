import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.pdf_extractor import extract_text_from_pdf
from app.services.docx_extractor import extract_text_from_docx
from app.services.ocr_extractor import extract_text_from_image
from app.services.report_analyzer import extract_medical_values
from app.models.predict_model import predict_disease
from app.explainability.shap_explainer import explain_prediction
from app.services.medical_summary import generate_enriched_summary

router = APIRouter(tags=["Report Analysis"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
MAX_SIZE_MB = 10


@router.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB}MB")

    # Write to random temp file — NOT named after original filename
    temp_path = os.path.join(tempfile.gettempdir(), f"pk_{uuid.uuid4().hex}{ext}")

    try:
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Extract text based on file type
        if ext == ".pdf":
            text = extract_text_from_pdf(temp_path)
        elif ext == ".docx":
            text = extract_text_from_docx(temp_path)
        elif ext in {".png", ".jpg", ".jpeg"}:
            text = extract_text_from_image(temp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

    finally:
        # Always delete the temp file — never persist patient documents
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Extract structured values from text
    extracted_values = extract_medical_values(text)

    # Determine which disease to predict based on what was extracted
    # Default: diabetes (most common use-case for this dataset)
    disease = "diabetes"
    patient_data = {
        "Pregnancies": extracted_values.get("Pregnancies", 0),
        "Glucose": extracted_values.get("Glucose", 100),
        "BloodPressure": extracted_values.get("BloodPressure", 70),
        "SkinThickness": extracted_values.get("SkinThickness", 20),
        "Insulin": extracted_values.get("Insulin", 80),
        "BMI": extracted_values.get("BMI", 25.0),
        "DiabetesPedigreeFunction": extracted_values.get("DiabetesPedigreeFunction", 0.5),
        "Age": extracted_values.get("Age", 45),
    }

    # Flag fields that were not found in the report (fell back to default)
    defaults_used = {k for k, v in extracted_values.items() if v == 0}

    prediction = predict_disease(disease, patient_data)
    explanation = explain_prediction(disease, patient_data)
    summary = generate_enriched_summary(
        disease=disease,
        risk_level=prediction["risk_level"],
        risk_score=prediction["risk_score"],
        top_shap_features=explanation["top_features"],
        feature_values=patient_data,
    )

    return {
        # NOTE: We do NOT return extracted_text — it may contain PII
        "extracted_fields": extracted_values,
        "defaults_used": list(defaults_used),
        "warning": (
            "Fields not found in report were set to population medians. "
            "For accurate results, provide all values manually."
            if defaults_used else None
        ),
        "prediction": prediction,
        "explanation": explanation,
        "summary": summary,
    }