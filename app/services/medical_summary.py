from typing import Dict, Any

# ─────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────

DISEASE_KNOWLEDGE = {
    "diabetes": {
        "name": "Type 2 Diabetes",
        "suggestions": {
            "high": [
                "Consult an endocrinologist immediately for HbA1c testing and treatment planning.",
                "Begin a low-glycemic diet: reduce white rice, bread, sugar-sweetened beverages.",
                "Aim for 150 minutes/week of moderate aerobic exercise (brisk walking, cycling).",
                "Monitor fasting blood glucose daily if possible.",
                "Lose 5–7% of body weight if overweight — this alone reduces diabetes risk by 58%.",
                "Avoid smoking and limit alcohol consumption.",
                "Check feet daily for sores or numbness (early neuropathy detection).",
            ],
            "medium": [
                "Schedule a fasting glucose test and HbA1c within the next 2–4 weeks.",
                "Reduce refined carbohydrate intake and increase dietary fiber.",
                "30 minutes of daily walking has measurable impact on insulin sensitivity.",
                "If BMI > 25, work with a dietitian for a sustainable weight-loss plan.",
                "Re-test blood glucose in 3 months.",
            ],
            "low": [
                "Maintain a balanced diet with low sugar and adequate fiber.",
                "Annual fasting glucose test is sufficient at this risk level.",
                "Maintain a healthy body weight and stay physically active.",
            ],
        },
        "recommended_tests": [
            "Fasting Plasma Glucose (FPG) — gold standard, requires 8h fast",
            "HbA1c (Glycated Hemoglobin) — reflects average glucose over 3 months",
            "Oral Glucose Tolerance Test (OGTT) — most sensitive for borderline cases",
            "Fasting Insulin level — assesses insulin resistance",
            "C-peptide test — measures how much insulin your body produces",
            "Urine microalbumin — checks for early kidney involvement",
            "Lipid panel — diabetes often co-occurs with dyslipidemia",
        ],
        "likely_causes_by_feature": {
            "Glucose": {
                "high": "Elevated blood glucose is the primary marker of impaired insulin function or resistance.",
                "normal": "Glucose is within range — not a significant contributor to this prediction."
            },
            "BMI": {
                "high": "Obesity (BMI > 30) is the strongest modifiable risk factor for Type 2 diabetes.",
                "normal": "Weight is not a primary risk driver here."
            },
            "Age": {
                "high": "Risk of Type 2 diabetes increases significantly after age 45.",
                "normal": "Age is not a dominant risk factor at this level."
            },
            "DiabetesPedigreeFunction": {
                "high": "Strong family history significantly elevates genetic risk.",
                "normal": "Family history is not a major contributor."
            },
            "Insulin": {
                "high": "Elevated insulin levels suggest compensatory hyperinsulinemia — early insulin resistance.",
                "normal": "Insulin levels are not flagging concern."
            },
        },
    },

    "heart": {
        "name": "Cardiovascular / Coronary Artery Disease",
        "suggestions": {
            "high": [
                "Seek cardiology consultation urgently — do not delay.",
                "If chest pain or shortness of breath occurs, go to the ER immediately.",
                "Begin cardiac medications only as prescribed (do not self-medicate with aspirin).",
                "Adopt a heart-healthy diet: Mediterranean diet, reduce saturated fat and sodium.",
                "Stop smoking immediately — it doubles the risk of heart attack.",
                "Monitor blood pressure daily; target < 130/80 mmHg.",
                "Limit alcohol to no more than 1 drink/day.",
                "Reduce stress — chronic stress directly increases cardiovascular risk.",
            ],
            "medium": [
                "Schedule a stress ECG (exercise tolerance test) with your doctor.",
                "Reduce dietary sodium to < 2,300 mg/day.",
                "Engage in moderate cardio 5 days/week — brisk walking, swimming.",
                "Get a lipid panel and fasting glucose test.",
                "Check blood pressure regularly.",
            ],
            "low": [
                "Annual cholesterol check is sufficient.",
                "Maintain aerobic activity and a low-fat diet.",
                "Do not smoke; maintain healthy blood pressure.",
            ],
        },
        "recommended_tests": [
            "Electrocardiogram (ECG/EKG) — detects arrhythmia and ischemia",
            "Echocardiogram — ultrasound of heart structure and function",
            "Exercise Stress Test (Treadmill test) — provokes ischemia under load",
            "Coronary CT Angiography (CTA) — images coronary arteries non-invasively",
            "Cardiac Catheterization — gold standard for coronary artery blockage",
            "Serum Troponin I/T — detects active heart muscle damage",
            "Lipid Panel (LDL, HDL, triglycerides)",
            "C-Reactive Protein (CRP) — inflammation marker tied to heart risk",
            "BNP / NT-proBNP — detects heart failure",
        ],
        "likely_causes_by_feature": {
            "age": {
                "high": "Age is a primary non-modifiable risk factor for coronary artery disease.",
                "normal": "Age is not a dominant factor here."
            },
            "trestbps": {
                "high": "Elevated resting blood pressure damages arterial walls over time, causing plaque buildup.",
                "normal": "Blood pressure is not a major contributor."
            },
            "chol": {
                "high": "High cholesterol promotes arterial plaque formation (atherosclerosis).",
                "normal": "Cholesterol is not flagging concern."
            },
            "cp": {
                "high": "Type of chest pain is a strong clinical indicator of cardiac involvement.",
                "normal": "Chest pain type not contributing significantly."
            },
        },
    },

    "kidney": {
        "name": "Chronic Kidney Disease (CKD)",
        "suggestions": {
            "high": [
                "Consult a nephrologist immediately for GFR testing and staging.",
                "Reduce dietary protein (excess protein accelerates CKD progression).",
                "Strictly limit potassium-rich foods (bananas, oranges, tomatoes) if potassium is elevated.",
                "Control blood pressure aggressively — target < 130/80 mmHg.",
                "Avoid NSAIDs (ibuprofen, naproxen) — they reduce kidney blood flow.",
                "Control blood sugar strictly if diabetic — diabetes is the leading cause of CKD.",
                "Stay well hydrated; track daily fluid intake.",
                "Get regular kidney function tests every 3 months.",
            ],
            "medium": [
                "Get a urine albumin-to-creatinine ratio (ACR) test.",
                "Keep blood pressure controlled.",
                "Reduce sodium to < 2g/day.",
                "Avoid contrast dye (CT scans with iodine) without prior nephrology consultation.",
            ],
            "low": [
                "Annual kidney function test (creatinine, eGFR) is advised.",
                "Stay hydrated and maintain a balanced diet.",
            ],
        },
        "recommended_tests": [
            "Serum Creatinine + eGFR — primary measure of kidney filtration rate",
            "Urine Albumin-to-Creatinine Ratio (ACR) — detects protein leak (key CKD marker)",
            "Blood Urea Nitrogen (BUN) — measures nitrogenous waste clearance",
            "Serum Electrolytes (Na, K, Cl, CO2) — CKD disrupts electrolyte balance",
            "Complete Blood Count (CBC) — CKD causes anemia",
            "Renal Ultrasound — checks for structural abnormalities",
            "Kidney Biopsy — definitive diagnosis for glomerular disease",
            "Serum Potassium — dangerous when elevated in CKD",
        ],
        "likely_causes_by_feature": {
            "bgr": {
                "high": "Elevated blood glucose is the single most common cause of CKD worldwide.",
                "normal": "Blood glucose not contributing to CKD risk here."
            },
            "hemo": {
                "high": "Anemia (low hemoglobin) is both a cause and consequence of CKD.",
                "normal": "Hemoglobin is not flagging concern."
            },
            "sc": {
                "high": "Elevated serum creatinine directly indicates impaired kidney filtration.",
                "normal": "Creatinine is within acceptable range."
            },
            "htn": {
                "high": "Hypertension is the second leading cause of CKD.",
                "normal": "Blood pressure not a major contributor."
            },
        },
    },

    "liver": {
        "name": "Liver Disease (Hepatic Dysfunction)",
        "suggestions": {
            "high": [
                "Consult a gastroenterologist or hepatologist immediately.",
                "Stop all alcohol consumption — alcohol is directly hepatotoxic.",
                "Avoid all acetaminophen (paracetamol) — toxic to a compromised liver.",
                "Do not take herbal supplements without hepatologist approval.",
                "Get vaccinated for Hepatitis A and B if not already immune.",
                "Adopt a liver-friendly diet: low-fat, high-fiber, no processed food.",
                "Avoid raw or undercooked shellfish (risk of hepatitis A).",
                "Discuss weight loss plan if BMI > 25 (fatty liver disease).",
            ],
            "medium": [
                "Schedule liver function tests (LFTs) and ultrasound.",
                "Reduce alcohol intake significantly.",
                "Get tested for Hepatitis B and C.",
                "Maintain healthy weight — obesity leads to NAFLD (non-alcoholic fatty liver).",
            ],
            "low": [
                "Annual liver function test is sufficient.",
                "Limit alcohol and maintain healthy weight.",
            ],
        },
        "recommended_tests": [
            "Liver Function Tests (LFTs): ALT, AST, ALP, GGT, Bilirubin, Albumin",
            "Hepatitis B Surface Antigen (HBsAg) — screens for Hepatitis B",
            "Hepatitis C Antibody (Anti-HCV) — screens for Hepatitis C",
            "Abdominal Ultrasound — detects fatty liver, cirrhosis, tumors",
            "FibroScan (Transient Elastography) — measures liver stiffness/fibrosis",
            "Prothrombin Time (PT/INR) — liver makes clotting factors; elevated in liver failure",
            "AFP (Alpha-fetoprotein) — tumor marker for hepatocellular carcinoma",
            "Liver Biopsy — gold standard for staging liver fibrosis",
        ],
        "likely_causes_by_feature": {
            "Total Bilirubin": {
                "high": "Elevated bilirubin indicates the liver is not clearing breakdown products — jaundice risk.",
                "normal": "Bilirubin is not contributing to this prediction."
            },
            "Sgot Aspartate Aminotransferase": {
                "high": "Elevated AST indicates liver cell damage (also elevated in heart damage).",
                "normal": "AST is not a major contributor."
            },
            "Alkphos Alkaline Phosphotase": {
                "high": "Elevated ALP suggests bile duct obstruction or liver inflammation.",
                "normal": "ALP not a significant contributor."
            },
        },
    },
}


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def generate_enriched_summary(
    disease: str,
    risk_level: str,
    risk_score: float,
    top_shap_features: list,
    feature_values: dict,
) -> dict:
    """
    Returns the full enriched prediction summary:
    - Suggestions
    - Recommended tests
    - Likely causes (based on top SHAP features)
    - Confidence interpretation
    - Graph data for frontend
    """
    kb = DISEASE_KNOWLEDGE.get(disease, {})
    if not kb:
        return {"error": f"No knowledge base for disease: {disease}"}

    risk_key = risk_level.lower()

    # Suggestions
    suggestions = kb["suggestions"].get(risk_key, kb["suggestions"]["low"])

    # Recommended tests (always return all, regardless of risk level)
    recommended_tests = kb["recommended_tests"]

    # Likely causes: for each top SHAP feature that increases risk,
    # look up the cause explanation
    likely_causes = []
    cause_kb = kb.get("likely_causes_by_feature", {})
    for feat in top_shap_features:
        if feat["direction"] == "increases_risk":
            feat_name = feat["feature"]
            if feat_name in cause_kb:
                cause_text = cause_kb[feat_name].get("high", "")
            else:
                cause_text = (
                    f"{feat_name} (value: {feat['value']:.2f}) is contributing "
                    f"positively to disease risk prediction."
                )
            if cause_text:
                likely_causes.append({
                    "feature": feat_name,
                    "value": feat["value"],
                    "explanation": cause_text,
                })

    # Confidence interpretation
    confidence_pct = round(risk_score * 100, 1)
    if risk_score > 0.7:
        confidence_text = (
            f"The model is {confidence_pct}% confident in a positive prediction. "
            "This is a high-confidence result — clinical follow-up is strongly advised."
        )
    elif risk_score > 0.4:
        confidence_text = (
            f"The model shows {confidence_pct}% probability of disease. "
            "This is a borderline result — further testing is recommended."
        )
    else:
        confidence_text = (
            f"The model shows {confidence_pct}% probability of disease. "
            "Risk appears low based on provided values."
        )

    # Graph data (ready for Recharts / Chart.js on frontend)
    # Bar chart: top 5 SHAP features with signed values
    shap_chart_data = [
        {
            "feature": f["feature"],
            "shap_value": round(f["shap"], 4),
            "direction": f["direction"],
        }
        for f in top_shap_features
    ]

    # Risk gauge data
    risk_gauge = {
        "score": round(risk_score, 4),
        "percentage": confidence_pct,
        "level": risk_level,
        "color": {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(
            risk_level, "#6b7280"
        ),
    }

    return {
        "disease_name": kb["name"],
        "risk_level": risk_level,
        "confidence": confidence_text,
        "risk_gauge": risk_gauge,
        "suggestions": suggestions,
        "recommended_tests": recommended_tests,
        "likely_causes": likely_causes,
        "shap_chart_data": shap_chart_data,
    }