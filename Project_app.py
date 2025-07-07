import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load model with cache
@st.cache_resource
def load_model():
    try:
        return load("model.joblib")
    except:
        st.error("Model loading failed. Check model.joblib exists.")
        st.stop()

model = load_model()
st.title("Breast Cancer Survival Predictor")
st.write("Predict 10-year survival risk for breast cancer patients.")

# Input mapping
INPUT_CONFIG = {
    "numerical": {
        "Age at Diagnosis": (20, 100, 50),
        "Cohort": (1, 20, 10),
        "Neoplasm Histologic Grade": (1, 3, 2),
        "Lymph nodes examined positive": (0, 50, 2),
        "Mutation Count": (0, 100, 5),
        "Nottingham prognostic index": (1.0, 10.0, 4.0),
        #"Overall Survival (Months)": (0, 300, 120),
        "Relapse Free Status (Months)": (0, 300, 120),
        "Tumor Size": (1, 150, 30),
        "Tumor Stage": (1, 4, 2)
    },
    "categorical": {
        "Type of Breast Surgery": ["Mastectomy", "Lumpectomy"],
        "Cancer Type": ["Breast Cancer", "Breast Sarcoma"],
        "Cancer Type Detailed": [
            "Breast Invasive Ductal Carcinoma", "Breast Mixed Ductal and Lobular Carcinoma", 
            "Breast Invasive Lobular Carcinoma", "Invasive Breast Carcinoma", 
            "Breast Invasive Mixed Mucinous Carcinoma", "Breast", "Breast Angiosarcoma", 
            "Metaplastic Breast Cancer"
        ],
        "Cellularity": ["High", "Moderate", "Low"],
        "Chemotherapy": ["Yes", "No"],
        "Pam50 + Claudin-low subtype": ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal", "NC"],
        "ER status measured by IHC": ["Positive", "Negative"],
        "ER Status": ["Positive", "Negative"],
        "HER2 status measured by SNP6": ["Positive", "Negative"],
        "HER2 Status": ["Positive", "Negative"],
        "Tumor Other Histologic Subtype": ["Ductal/NST", "Lobular", "Mixed"],
        "Hormone Therapy": ["Yes", "No"],
        "Inferred Menopausal State": ["Post", "Pre"],
        "Integrative Cluster": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "Primary Tumor Laterality": ["Left", "Right"],
        "Oncotree Code": ["IDC", "ILC", "MDL", "Other"],
        "PR Status": ["Positive", "Negative"],
        "Radio Therapy": ["Yes", "No"],
        "Relapse Free Status": ["Relapsed", "Not Relapsed"],
        "3-Gene classifier subtype": ["ER+/HER2- Low Prolif", "ER+/HER2- High Prolif", "ER-/HER2-", "HER2+"],
    }
}

# Dynamic input creation
with st.sidebar:
    st.header("Patient Data")
    inputs = {}
    
    # Numerical inputs
    for feat, (min_val, max_val, default) in INPUT_CONFIG["numerical"].items():
        inputs[feat] = st.number_input(feat, min_val, max_val, default)
    
    # Categorical inputs
    for feat, options in INPUT_CONFIG["categorical"].items():
        inputs[feat] = st.selectbox(feat, options)

# Feature encoding
feature_map = {
    "Mastectomy": 1, "Lumpectomy": 0,
    "Breast Cancer": 0, "Breast Sarcoma": 1,
    "Yes": 1, "No": 0,
    "Positive": 1, "Negative": 0,
    "Post": 0, "Pre": 1,
    "Left": 0, "Right": 1,
    "Relapsed": 1, "Not Relapsed": 0,
    "Deceased": 0, "Living": 1,
    "Ductal/NST": 0, "Lobular": 1, "Mixed": 2,
    "IDC": 0, "ILC": 1, "MDL": 2, "Other": 3
}

# Convert categorical inputs
encoded_inputs = []
for feat in INPUT_CONFIG["categorical"]:
    encoded_inputs.append(feature_map.get(inputs[feat], 0))  # Default to 0 if not in feature_map

# Convert numerical inputs
numerical_inputs = [inputs[feat] for feat in INPUT_CONFIG["numerical"]]

# Combine features into a NumPy array
features = np.array(numerical_inputs + encoded_inputs).reshape(1, -1)

# Prediction and display
if st.sidebar.button("Predict"):
    try:
        proba = model.predict_proba(features)[0]
        prediction = model.predict(features)[0]
        
        survival_status = "Living ✅" if prediction == 1 else "Deceased ⚠️"
        confidence = f"{max(proba) * 100:.1f}%"

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction")
            st.markdown(f"**{survival_status}**")
            st.metric("Confidence", confidence)
            
        with col2:
            st.subheader("Interpretation")
            if prediction == 1:
                st.write("The patient is predicted to survive. Regular follow-ups are recommended.")
            else:
                st.write("High risk of mortality. Consider immediate medical consultation.")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

st.caption("For clinical research use only - v3.0.1")

