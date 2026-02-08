from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    X = np.array([[
        data["age"],
        data["gender"],
        data["bmi"],
        data["sbp"],
        data["dbp"],
        data["fpg"],
        data["ffpg"],
        data["chol"],
        data["tri"],
        data["hdl"],
        data["ldl"],
        data["smoking"],
        data["drinking"],
        data["family_history"]
    ]])

    prob = model.predict_proba(X)[0][1]
    diagnosis = "Diabetes Detected" if prob >= 0.5 else "No Diabetes Detected"

    return jsonify({
        "probability": round(prob, 3),
        "diagnosis": diagnosis
    })

if __name__ == "__main__":
    app.run(debug=True)
