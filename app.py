from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and features
model = joblib.load("breast_cancer_model.pkl")
features = joblib.load("features.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            data = [float(request.form[feature]) for feature in features if feature not in ['target', 'cluster']]
            prediction = model.predict([data])[0]
            result = "Malignant" if prediction == 0 else "Benign"
        except:
            result = "Invalid input. Please enter all fields correctly."
        return render_template("index.html", result=result, features=features)
    return render_template("index.html", features=features)

if __name__ == "__main__":
    app.run(debug=True)
