from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Check if model file exists, otherwise create a simple model
model_path = "breast_cancer_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Loaded existing model")
else:
    # Fallback - create a simple model if the saved one isn't available
    # This helps during deployment if the model file is missing
    print("Model not found, creating a simple substitute model")
    from sklearn.datasets import load_breast_cancer

    # Load the dataset
    breast_cancer = load_breast_cancer()
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y = breast_cancer.target

    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the model for future use
    joblib.dump(model, model_path)

    # Save feature names
    features = X.columns.tolist()
    joblib.dump(features, "features.pkl")
    print(f"Created and saved new model with {len(features)} features")

# Load feature names
if os.path.exists("features.pkl"):
    features = joblib.load("features.pkl")
else:
    # Fallback - use feature names from breast cancer dataset
    from sklearn.datasets import load_breast_cancer

    features = load_breast_cancer().feature_names
    joblib.dump(features, "features.pkl")
    print("Created features.pkl from dataset")


@app.route("/")
def home():
    return render_template("index.html", features=features)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values for each feature
        input_data = []
        for feature in features:
            value = request.form.get(feature, 0)
            # Convert to float, default to 0 if conversion fails
            try:
                input_data.append(float(value))
            except ValueError:
                input_data.append(0.0)

        # Make prediction
        prediction = model.predict([input_data])[0]
        probability = model.predict_proba([input_data])[0][1]

        # Format results
        result = "Benign (Non-cancerous)" if prediction == 1 else "Malignant (Cancerous)"
        prob_percentage = f"{probability * 100:.2f}%" if prediction == 1 else f"{(1 - probability) * 100:.2f}%"

        return render_template("index.html",
                               features=features,
                               prediction=result,
                               probability=prob_percentage,
                               input_values={feature: input_data[i] for i, feature in enumerate(features)})

    except Exception as e:
        error_message = f"Prediction error: {str(e)}"
        print(error_message)
        return render_template("index.html", features=features, error=error_message)


if __name__ == "__main__":
    app.run(debug=True)