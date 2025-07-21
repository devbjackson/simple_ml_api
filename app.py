from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# Path to our trained model
MODEL_PATH = 'model.joblib'
model = None # Initialize model as None

# Load the model when the application starts
# This is better than loading it on every request for performance
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model '{MODEL_PATH}' loaded successfully.")
    else:
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run train.py first.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return "Welcome to the Simple ML API! Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure train.py was run and model.joblib exists.'}), 500

    try:
        # Get data from POST request (JSON format)
        data = request.get_json(force=True)
        # Expecting 'features' as a list in the JSON body, e.g., {"features": [15]}
        features = np.array(data['features']).reshape(-1, 1)

        # Make prediction
        prediction = model.predict(features)

        # Return prediction
        return jsonify({'prediction': prediction.tolist()})

    except KeyError:
        return jsonify({'error': 'Invalid JSON format. Please send {"features": [value]}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For development, Flask's built-in server is fine.
    # In production, you'd use a WSGI server like Gunicorn.
    app.run(host='0.0.0.0', port=5000, debug=True)