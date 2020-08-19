from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path

# Paths
model_path = Path(
    Path().absolute(), 
    'models'
    )

# Load model and feature names
model = load(Path(model_path, 'final_model.pkl'))
feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# build app
app = Flask(__name__)

# To test main page
@app.route("/")
def hello():
    return "Hello, World!"

# Define App routes
@app.route('/predict', methods=['POST'])

def predict(): 
    # if request.method == 'POST':
    dict_ = {}

    # Build dictionary of features
    for feature in feature_names:
        dict_[feature] = [request.form[feature]]

    # Convert to df
    df = pd.DataFrame(dict_)

    # Prediction
    pred = model.predict(df)

    return jsonify({"prediction": pred[0]})


@app.route('/predict_proba', methods=['POST'])

def predict_proba(): 
    # if request.method == 'POST':
    dict_ = {}

    # Build dictionary of features
    for feature in feature_names:
        dict_[feature] = [request.form[feature]]

    # Convert to df
    df = pd.DataFrame(dict_)

    # Prediction
    pred = model.predict_proba(df)

    # Convert to dict
    pred_dict = {}
    for i in range(len(class_names)):
        pred_dict[class_names[i]] = pred[0][i]

    return jsonify(pred_dict)


if __name__ == '__main__':
    app.run(host='0.0.0.0')