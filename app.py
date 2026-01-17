from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        features = [
            float(request.form["radius"]),
            float(request.form["texture"]),
            float(request.form["perimeter"]),
            float(request.form["area"]),
            float(request.form["symmetry"]),
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        result = model.predict(features_scaled)
        prediction = "Benign" if result[0] == 1 else "Malignant"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
