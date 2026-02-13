from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

model = joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    values = {
        "Age": data["age"],
        "BMI": data["bmi"],
        "SBP": data["sbp"],
        "DBP": data["dbp"],
        "FPG": data["fpg"],
        "FFPG": data["ffpg"],
        "Chol": data["chol"],
        "Tri": data["tri"],
        "HDL": data["hdl"],
        "LDL": data["ldl"]
    }

    # ML Prediction
    features = np.array([[ 
        data["age"], data["gender"], data["bmi"], data["sbp"],
        data["dbp"], data["fpg"], data["ffpg"], data["chol"],
        data["tri"], data["hdl"], data["ldl"],
        data["smoking"], data["drinking"], data["family_history"]
    ]])

    prob = model.predict_proba(features)[0][1]
    diagnosis = "Diabetes Detected" if prob >= 0.5 else "No Diabetes Detected"

    # ---- Clinical Cutoffs ----
    cutoffs = {
        "BMI": 25,
        "SBP": 130,
        "DBP": 90,
        "FPG": 5.6,
        "FFPG": 7.8,
        "Chol": 5.2,
        "Tri": 1.7,
        "LDL": 2.6
    }

    protective_cutoffs = {
        "HDL": 1.55
    }

    features_plot = []
    deviations = []
    colors = []

    for key in values:
        if key in protective_cutoffs:
            cutoff = protective_cutoffs[key]
            if values[key] >= cutoff:
                deviations.append(-(values[key] - cutoff))
                colors.append("blue")
            else:
                deviations.append(cutoff - values[key])
                colors.append("red")
            features_plot.append(key)

        elif key in cutoffs:
            cutoff = cutoffs[key]
            if values[key] >= cutoff:
                deviations.append(values[key] - cutoff)
                colors.append("red")
            else:
                deviations.append(-(cutoff - values[key]))
                colors.append("blue")
            features_plot.append(key)

    # Sort by strongest deviation
    sorted_data = sorted(zip(features_plot, deviations, colors),
                         key=lambda x: abs(x[1]), reverse=True)

    features_plot, deviations, colors = zip(*sorted_data)

    # ---- Plot ----
    plt.figure(figsize=(6,4))
    plt.barh(features_plot, deviations, color=colors)
    plt.axvline(0)
    plt.title("Clinical Risk Interpretation")
    plt.xlabel("Deviation from Clinical Cutoff")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        "probability": round(float(prob), 3),
        "diagnosis": diagnosis,
        "plot": plot_url
    })

if __name__ == "__main__":
    app.run(debug=True, port=10000)
