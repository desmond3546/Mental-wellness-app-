from flask import Flask, render_template, request
import traceback
from model_utils import preprocess_and_predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_dict = request.form.to_dict()
        print("🧩 FORM DATA RECEIVED:", form_dict)

        def safe_get(key, default="unknown"):
            val = form_dict.get(key)
            if val is None or str(val).strip() == "":
                return default
            return str(val).strip()

        # 🧠 Normalize keys to match model expectations
        form_data = {
            "Gender": safe_get("gender"),
            "Country": safe_get("country"),
            "Occupation": safe_get("occupation"),
            "self_employed": "no",  # not in form
            "family_history": safe_get("family_history", "no"),
            "Days_Indoors": safe_get("days_indoors", "15-30 days"),
            "Growing_Stress": safe_get("growing_stress", "no"),
            "Changes_Habits": safe_get("changes_habits", "no"),
            "Mental_Health_History": "no",  # not in form
            "Mood_Swings": safe_get("mood_swings", "medium"),
            "Coping_Struggles": safe_get("coping_struggles", "no"),
            "Work_Interest": safe_get("work_interest", "no"),
            "Social_Weakness": "no",  # not in form
            "mental_health_interview": safe_get("mental_health_interview", "no"),
            "care_options": safe_get("care_options", "no"),
            "Year": 2014,
            "Month": 8,
            "Weekday": "wednesday",
            "Hour": 11
        }

        # ✅ Call model
        prediction, confidence = preprocess_and_predict(form_data)

        if prediction is None:
            return render_template("error.html",
                                   message="Prediction Failed — please check your inputs.",
                                   confidence=0)

        return render_template("result.html",
                               prediction=prediction,
                               confidence=round(confidence * 100, 2))

    except Exception as e:
        print("❌ Error in /predict route:", e)
        print(traceback.format_exc())
        return render_template("error.html",
                               message=f"Prediction Failed: {str(e)}",
                               confidence=0)


if __name__ == "__main__":
    app.run(debug=True)
