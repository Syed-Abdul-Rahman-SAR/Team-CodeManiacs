from ml_service.symptom_predictor import predict_disease_from_symptoms
from ml_service.severity_lookup import get_disease_severity

def triage_patient(symptoms=None, disease=None):
    """
    symptoms: list of symptoms (optional)
    disease: disease name (optional)

    One of them MUST be provided.
    """

    # Step 1: Decide disease source
    if disease is None:
        if symptoms is None or len(symptoms) == 0:
            raise ValueError("Either symptoms or disease must be provided")

        # Disease comes from ML model
        disease_name = predict_disease_from_symptoms(symptoms)

    else:
        # Disease comes directly from frontend
        disease_name = disease.strip().lower()

    # Step 2: Find severity
    severity = get_disease_severity(disease_name)

    return {
        "disease": disease_name,
        "severity": severity
    }


# =========================
# Local Testing
# =========================
if __name__ == "__main__":
    print(triage_patient(symptoms=["fever", "headache"]))
    print(triage_patient(disease="heart attack"))
