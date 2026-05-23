import re

def extract_medical_features(text):

    features = {}

    text = text.lower()

    features["fever"] = int("fever" in text)
    features["chest_pain"] = int("chest pain" in text)
    features["diabetes"] = int("diabetes" in text)
    features["hypertension"] = int("hypertension" in text)

    glucose_match = re.search(r'glucose\\D+(\\d+)', text)

    if glucose_match:
        features["glucose"] = int(glucose_match.group(1))
    else:
        features["glucose"] = 0

    return features