import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/symptom_disease_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
feature_names = joblib.load("models/feature_names.pkl")

def predict_disease(symptoms_present):
    input_vector = np.zeros(len(feature_names))

    for symptom in symptoms_present:
        symptom = symptom.strip().lower().replace(" ", "_")
        if symptom in feature_names:
            idx = feature_names.index(symptom)
            input_vector[idx] = 1

    input_df = pd.DataFrame([input_vector], columns=feature_names)

    pred_class = model.predict(input_df)[0]
    disease_name = label_encoder.inverse_transform([pred_class])[0]

    return disease_name


if __name__ == "__main__":
    symptoms = ["fever", "headache", "neck_weakness"]
    print("Predicted disease:", predict_disease(symptoms))
