import re

def extract_medical_values(text):

    extracted = {}

    text = text.lower()

    # Debug: log the extracted text so we can diagnose failures
    print("=" * 60)
    print("EXTRACTED TEXT FOR ANALYSIS:")
    print(text[:2000])
    print("=" * 60)

    # Flexible separator: matches colon, whitespace, equals, dashes, pipes, commas, and tabs
    # This handles formats like "BP: 120", "BP = 120", "BP - 120", "BP  120", "BP\t120"
    SEP = r'[\s:=\-–—\|,]*\s*'

    # Number pattern for integers and decimals
    NUM = r'(\d+(?:\.\d+)?)'

    patterns = {
        "Glucose": [
            r'(?:fasting\s+)?blood\s+sugar' + SEP + NUM,
            r'(?:fasting\s+)?(?:blood\s+)?(?:plasma\s+)?glucose' + SEP + NUM,
            r'\bfbs\b' + SEP + NUM,
            r'\brbs\b' + SEP + NUM,
        ],

        "HbA1c": [
            r'hba1c' + SEP + NUM,
            r'hb\s*a1c' + SEP + NUM,
            r'a1c' + SEP + NUM,
            r'glycated\s+h[ae]moglobin' + SEP + NUM,
            r'glycosylated\s+h[ae]moglobin' + SEP + NUM,
        ],

        "BloodPressure": [
            r'(?:systolic\s+)?blood\s+pressure' + SEP + NUM,
            r'\bbp\b' + SEP + NUM,
            r'systolic' + SEP + NUM,
            r'\bsbp\b' + SEP + NUM,
            # Match "120/80" format and capture systolic (first number)
            r'\bbp\b' + SEP + r'(\d+)\s*/\s*\d+',
            r'blood\s+pressure' + SEP + r'(\d+)\s*/\s*\d+',
        ],

        "BMI": [
            r'\bbmi\b' + SEP + r'([0-9]+(?:\.[0-9]+)?)',
            r'body\s+mass\s+index' + SEP + r'([0-9]+(?:\.[0-9]+)?)',
        ],
        "Age": [
            # Handle "Age / Gender  41 / Male" table format
            r'age\s*/\s*gender' + SEP + r'(\d+)',
            r'\bage\b' + SEP + r'(\d+)',
            r'(\d+)\s*years?\s*old',
            r'age' + SEP + r'(\d+)\s*(?:yrs?|years?)',
            r'patient\s+age' + SEP + r'(\d+)',
        ],

        "Cholesterol": [
            # Match plain "cholesterol" but NOT "hdl cholesterol" or "ldl cholesterol"
            r'(?<!hdl\s)(?<!ldl\s)(?<!hdl )(?<!ldl )total\s+cholesterol' + SEP + NUM,
            r'(?<!\w)cholesterol' + SEP + NUM,
        ],

        "Triglycerides": [
            r'triglycerides?' + SEP + NUM,
            r'\btg\b' + SEP + NUM,
        ],

        "HDL": [
            r'hdl\s*(?:cholesterol)?' + SEP + NUM,
            r'high\s+density\s+lipoprotein' + SEP + NUM,
        ],

        "LDL": [
            r'ldl\s*(?:cholesterol)?' + SEP + NUM,
            r'low\s+density\s+lipoprotein' + SEP + NUM,
        ],

        "Insulin": [
            r'(?:serum\s+)?insulin' + SEP + NUM,
            r'fasting\s+insulin' + SEP + NUM,
        ],
        "SkinThickness": [
            r'skin\s*thickness' + SEP + NUM,
            r'triceps\s*(?:skin\s*fold)?' + SEP + NUM,
        ],
        "DiabetesPedigreeFunction": [
            r'diabetes\s+pedigree\s*(?:function)?' + SEP + NUM,
            r'\bdpf\b' + SEP + NUM,
            r'pedigree\s*(?:function)?' + SEP + NUM,
        ],
        "Pregnancies": [
            r'pregnanc(?:y|ies)' + SEP + NUM,
            r'gravida' + SEP + NUM,
        ],
        # Liver-specific
        "Bilirubin": [
            r'total\s+bilirubin' + SEP + NUM,
            r'\bbilirubin\b' + SEP + NUM,
            r'\btbil\b' + SEP + NUM,
        ],
        "ALT": [
            r'\balt\b\s*(?:\(sgpt\))?' + SEP + NUM,
            r'\bsgpt\b\s*(?:\(alt\))?' + SEP + NUM,
            r'sgpt\s*/\s*alt' + SEP + NUM,
            r'alt\s*/\s*sgpt' + SEP + NUM,
            r'alanine\s+(?:amino\s*)?transaminase' + SEP + NUM,
            r'alamine\s+aminotransferase' + SEP + NUM,
        ],
        "AST": [
            r'\bast\b\s*(?:\(sgot\))?' + SEP + NUM,
            r'\bsgot\b\s*(?:\(ast\))?' + SEP + NUM,
            r'sgot\s*/\s*ast' + SEP + NUM,
            r'ast\s*/\s*sgot' + SEP + NUM,
            r'aspartate\s+(?:amino\s*)?transaminase' + SEP + NUM,
        ],
        # Kidney-specific
        "Creatinine": [
            r'(?:serum\s+)?creatinine' + SEP + NUM,
            r'\bcreat\b' + SEP + NUM,
        ],
        "Hemoglobin": [
            r'h[ae]moglobin' + SEP + NUM,
            r'\bhgb\b' + SEP + NUM,
            r'\bhb\b' + SEP + NUM,
        ],
        # Vitamins
        "VitaminD": [
            r'vitamin\s*d\b' + SEP + NUM,
            r'vit\s*\.?\s*d\b' + SEP + NUM,
            r'25-hydroxy\s*vitamin\s*d' + SEP + NUM,
        ],
        "VitaminB12": [
            r'vitamin\s*b\s*12' + SEP + NUM,
            r'vit\s*\.?\s*b\s*12' + SEP + NUM,
            r'\bb12\b' + SEP + NUM,
            r'cobalamin' + SEP + NUM,
        ],
    }
    for feature, regex_list in patterns.items():
        extracted[feature] = 0
        for pattern in regex_list:
            match = re.search(pattern, text)
            if match:

                value = match.group(1)

                if "." in value:
                    extracted[feature] = float(value)
                else:
                    extracted[feature] = int(value)

                print(f"  ✓ {feature} = {extracted[feature]} (matched: {pattern[:50]}...)")
                break
        if extracted[feature] == 0:
            print(f"  ✗ {feature} = NOT FOUND")

    return extracted