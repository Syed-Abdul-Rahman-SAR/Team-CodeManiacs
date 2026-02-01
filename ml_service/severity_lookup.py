import pandas as pd

# =========================
# Load Excel files ONCE
# =========================
emergency_df = pd.read_excel("data/raw/emergency.xlsx")
medium_df = pd.read_excel("data/raw/medium.xlsx")
normal_df = pd.read_excel("data/raw/normal.xlsx")

# Normalize disease names
emergency_set = set(emergency_df["Disease"].str.strip().str.lower())
medium_set = set(medium_df["Disease"].str.strip().str.lower())
normal_set = set(normal_df["Disease"].str.strip().str.lower())

# =========================
# MAIN FUNCTION
# =========================
def get_disease_severity(disease_name):
    """
    disease_name: string
    returns: EMERGENCY / MEDIUM / NORMAL
    """

    disease = disease_name.strip().lower()

    if disease in emergency_set:
        return "EMERGENCY"
    elif disease in medium_set:
        return "MEDIUM"
    elif disease in normal_set:
        return "NORMAL"
    else:
        # fallback (safe default)
        return "NORMAL"


# =========================
# Local Test
# =========================
if __name__ == "__main__":
    tests = [
        "heart attack",
        "hypertension",
        "panic disorder",
        "unknown disease"
    ]

    for d in tests:
        print(d, "→", get_disease_severity(d))
