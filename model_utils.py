import pandas as pd
import numpy as np
import joblib
import pickle
import traceback

# =========================
# Load model and encoders once (fast!)
# =========================
stack_model_1 = joblib.load("stack_model_1.pkl")
mood_encoder = joblib.load("mood_encoder.pkl")
days_encoder = joblib.load("days_encoder.pkl")
scaler = joblib.load("scaler.pkl")
final_columns = joblib.load("final_columns.pkl")

with open("binary_map.pkl", "rb") as f:
    binary_map = pickle.load(f)

# =========================
# Cached helper data
# =========================
weekday_map = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
}

categorical_fields = [
    "Gender", "Country", "Occupation", "Work_Interest",
    "mental_health_interview", "self_employed", "family_history",
    "Coping_Struggles", "care_options"
]

numeric_cols = [
    "Days_Indoors", "Growing_Stress", "Changes_Habits",
    "Mental_Health_History", "Mood_Swings", "Social_Weakness",
    "Year", "Month", "Weekday", "Hour", "Month_sin", "Month_cos",
    "Weekday_sin", "Weekday_cos"
]


# =========================
# Safe encoder helper (optimized)
# =========================
def safe_encode(encoder, value, default):
    try:
        if value not in encoder.categories_[0]:
            value = default
        return encoder.transform([[value]])[0]
    except Exception:
        return encoder.transform([[default]])[0]


# =========================
# Main prediction function
# =========================
def preprocess_and_predict(form_data):
    try:
        # 1️⃣ Convert form input to DataFrame
        new_data = pd.DataFrame([form_data])

        # 2️⃣ Compute time features (lightweight)
        new_data["Weekday"] = new_data["Weekday"].map(weekday_map).fillna(0).astype(int)
        new_data["Month_sin"] = np.sin(2 * np.pi * new_data["Month"] / 12)
        new_data["Month_cos"] = np.cos(2 * np.pi * new_data["Month"] / 12)
        new_data["Weekday_sin"] = np.sin(2 * np.pi * new_data["Weekday"] / 7)
        new_data["Weekday_cos"] = np.cos(2 * np.pi * new_data["Weekday"] / 7)

        # 3️⃣ Encodings (safer + faster)
        new_data["Mood_Swings"] = safe_encode(mood_encoder, new_data["Mood_Swings"].iloc[0], "medium")
        new_data["Days_Indoors"] = safe_encode(days_encoder, new_data["Days_Indoors"].iloc[0], "15-30 days")

        # 4️⃣ Binary mappings
        for col in ["Growing_Stress", "Changes_Habits", "Mental_Health_History", "Social_Weakness"]:
            new_data[col] = new_data[col].map(binary_map).fillna(0).astype(int)

        # 5️⃣ Initialize model input (template copied once)
        X_new = pd.DataFrame(np.zeros((1, len(final_columns))), columns=final_columns)

        # 6️⃣ Insert numeric / cyclic values
        for col in numeric_cols:
            if col in X_new.columns:
                X_new[col] = new_data[col].values

        # 7️⃣ One-hot categorical encoding
        for field in categorical_fields:
            val = str(new_data[field].iloc[0]).strip().lower().replace(" ", "_")
            col_name = f"{field}_{val}"
            if col_name in X_new.columns:
                X_new[col_name] = 1

        # 8️⃣ Scale + predict
        X_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns)

        if hasattr(stack_model_1, "predict_proba"):
            proba = stack_model_1.predict_proba(X_scaled)[0]
            prediction = int(np.argmax(proba))
            confidence = float(np.max(proba))
        else:
            prediction = int(stack_model_1.predict(X_scaled)[0])
            confidence = 0.5

        return prediction, confidence

    except Exception as e:
        print("❌ Prediction Error:", e)
        print(traceback.format_exc())
        return None, 0.0
