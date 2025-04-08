from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the trained model
model = joblib.load("linear_regression_model.pkl")

# Sample route for the homepage
@app.route('/')
def home():
    return "🏡 Boston Housing Price Prediction API is Running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()

    # Convert data to a DataFrame (you can adjust the input format as needed)
    user_input = pd.DataFrame([data])

    # Predict the house price
    prediction = model.predict(user_input)

    # Return the prediction as a JSON response
    return jsonify({"predicted_price": prediction[0] * 1000})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

