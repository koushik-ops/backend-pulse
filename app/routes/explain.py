from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.explainability.shap_explainer import explain_prediction

router = APIRouter(prefix="/explain", tags=["Explainability"])


class ExplainRequest(BaseModel):
    disease: str  # 'diabetes', 'heart', 'kidney', 'liver'
    features: dict  # {feature_name: value}


@router.post("/")
def explain(req: ExplainRequest):
    allowed = {"diabetes", "heart", "kidney", "liver"}
    if req.disease not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown disease. Choose from: {allowed}"
        )
    try:
        return explain_prediction(req.disease, req.features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))