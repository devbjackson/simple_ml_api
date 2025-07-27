# Simple ML API

A simple machine learning API built with Flask that serves predictions from a Linear Regression model.

## Overview

This project demonstrates a basic machine learning pipeline with model training and serving via REST API. It uses scikit-learn to train a Linear Regression model on synthetic data and Flask to create a web API for making predictions.

## Features

- **Model Training**: Train a Linear Regression model on synthetic data
- **REST API**: Flask-based API for serving predictions
- **Error Handling**: Comprehensive error handling for API requests
- **Model Persistence**: Save and load trained models using joblib

## Project Structure

```
simple_ml_api/
├── app.py              # Flask API server
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── model.joblib        # Trained model (generated after running train.py)
└── README.md          # This file
```

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd simple_ml_api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train.py
   ```
   This will:
   - Generate synthetic training data
   - Train a Linear Regression model
   - Save the model as `model.joblib`

4. **Start the API server**
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:5000`

## Usage

### API Endpoints

#### GET `/`
Welcome endpoint that returns a greeting message.

**Response:**
```
Welcome to the Simple ML API! Use /predict to get predictions.
```

#### POST `/predict`
Make predictions using the trained model.

**Request Format:**
```json
{
    "features": [value]
}
```

**Example Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5]}'
```

**Example Response:**
```json
{
    "prediction": [6.0]
}
```

### Example Usage with Python

```python
import requests
import json

# Make a prediction
url = "http://localhost:5000/predict"
data = {"features": [7]}
response = requests.post(url, json=data)
result = response.json()
print(f"Prediction: {result['prediction'][0]}")
```

## Model Details

- **Algorithm**: Linear Regression
- **Training Data**: Synthetic data with 10 samples
- **Features**: Single feature (X values from 1 to 10)
- **Target**: Approximately linear relationship with some noise
- **Model File**: Saved as `model.joblib` using joblib

The trained model learns a linear relationship: `y ≈ 1.0 * x + 1.0`

## Error Handling

The API includes comprehensive error handling for:

- **Model not found**: Returns 500 error if `model.joblib` doesn't exist
- **Invalid JSON format**: Returns 400 error for malformed requests
- **Missing features**: Returns 400 error if 'features' key is missing
- **General exceptions**: Returns 500 error with descriptive message

## Development

### Running in Development Mode

The app runs in debug mode by default when using `python app.py`. This enables:
- Auto-reload on code changes
- Detailed error messages
- Debug toolbar

### Production Deployment

For production deployment, consider using a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Dependencies

- **Flask**: Web framework for the API
- **scikit-learn**: Machine learning library
- **joblib**: Model serialization
- **numpy**: Numerical computing
- **gunicorn**: WSGI server (for production)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Future Improvements

- Add data validation and preprocessing
- Implement multiple model support
- Add model versioning
- Include unit tests
- Add logging and monitoring
- Support batch predictions
- Add model retraining endpoints
- Implement authentication and rate limiting
