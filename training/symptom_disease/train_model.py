import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("data/raw/symptom_disease.csv")

print("Dataset shape:", df.shape)
print(df.head())

# =========================
# 2. Clean Column Names
# =========================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('[^a-z0-9_]', '', regex=True)
)

# =========================
# 3. Split X and y
# =========================
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 4. Encode Labels
# =========================
le = LabelEncoder()
y_enc = le.fit_transform(y)

print("Total diseases:", len(le.classes_))

# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    random_state=42
)

# =========================
# 6. MEMORY-SAFE MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=50,          # VERY IMPORTANT: small
    max_depth=20,
    min_samples_leaf=10,
    n_jobs=1,                 # 🔴 SINGLE CORE (CRITICAL)
    random_state=42
)

print("Training started...")
model.fit(X_train, y_train)
print("Training completed.")

# =========================
# 7. Evaluation
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# =========================
# 8. Save Artifacts
# =========================
joblib.dump(model, "models/symptom_disease_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

print("Model, encoder, and feature names saved.")
