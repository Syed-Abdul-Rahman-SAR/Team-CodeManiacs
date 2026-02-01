import joblib
import numpy as np
import pandas as pd

# Load trained model artifacts
model = joblib.load("models/symptom_disease_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
feature_names = joblib.load("models/feature_names.pkl")

def predict_disease_from_symptoms(symptoms_list):
    """
    symptoms_list: list of selected symptoms from frontend/backend
    example: ["fever", "headache", "neck_weakness"]
    """

    # Create zero vector (all symptoms = 0)
    input_vector = np.zeros(len(feature_names))

    # Set selected symptoms to 1
    for symptom in symptoms_list:
        symptom = symptom.strip().lower().replace(" ", "_")
        if symptom in feature_names:
            idx = feature_names.index(symptom)
            input_vector[idx] = 1

    # Convert to DataFrame with proper feature names
    input_df = pd.DataFrame([input_vector], columns=feature_names)

    # Predict disease
    pred_class = model.predict(input_df)[0]
    disease_name = label_encoder.inverse_transform([pred_class])[0]

    return disease_name

if __name__ == "__main__":
    test_symptoms = ["fever", "headache"]
    print(predict_disease_from_symptoms(test_symptoms))
