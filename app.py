from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models
model_path = os.path.join("models", "decision_tree_model.pkl")
scaler_path = os.path.join("models", "scaler.pkl")
pca_path = os.path.join("models", "pca_model.pkl")

DT = joblib.load(model_path)
sc = joblib.load(scaler_path)
pca = joblib.load(pca_path)

# Features with display names and range hints
features = [
    {"name": "Specific gravity", "range": "(0.0–5.0)"},
    {"name": "Albumin", "range": "(0.0–5.0)"},
    {"name": "Sugar", "range": "(0.0–5.0)"},
    {"name": "Blood glucose", "range": "(20–500)"},
    {"name": "Serum creatinine", "range": "(0.0–50.0)"},
    {"name": "Potassium", "range": "(2.0–8.0)"},
    {"name": "Packed cell volume", "range": "(0.0–60)"},
    {"name": "White blood cell", "range": "(2000–3000)"},
    {"name": "Red Blood cell", "range": "(2.0–10)"}
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    prediction_class = None
    error = None

    label_map = {
        "ckd": ("Chronic Kidney Disease (CKD) Detected", "ckd"),
        "notckd": ("No Chronic Kidney Disease (Healthy)", "notckd"),
    }

    if request.method == 'POST':
        try:
            input_data = []

            for feat in features:
                input_data.append(float(request.form[feat["name"]]))

            diabetes_val = request.form['diabetes']
            if diabetes_val == 'no':
                input_data.extend([1.0, 0.0])
            elif diabetes_val == 'yes':
                input_data.extend([0.0, 1.0])
            else:
                raise ValueError("Invalid diabetes input")

            raw_input = np.array(input_data).reshape(1, -1)
            scaled_input = sc.transform(raw_input)
            pca_input = pca.transform(scaled_input)
            raw_prediction = str(DT.predict(pca_input)[0]).strip().lower()
            prediction, prediction_class = label_map.get(
                raw_prediction, (raw_prediction, "neutral")
            )

        except ValueError:
            error = "Please enter valid numerical values and select diabetes status."

    return render_template(
        'index.html',
        features=features,
        prediction=prediction,
        prediction_class=prediction_class,
        error=error,
    )

if __name__ == '__main__':
    app.run(debug=True)

