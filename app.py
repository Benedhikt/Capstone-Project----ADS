from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("linear_regression_model.pkl", "rb"))

@app.route('/')
def home():
    return "🏡 Boston Housing Price Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # Getting data from the request
    input_data = np.array(data['features']).reshape(1, -1)  # Features to model input

    prediction = model.predict(input_data)  # Predict price

    # Return prediction
    return jsonify({'prediction': prediction[0] * 1000})

if __name__ == '__main__':
    app.run(debug=True)

