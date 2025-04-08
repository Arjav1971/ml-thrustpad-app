from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # For .h5 models

app = Flask(__name__)
CORS(app)

input_columns = joblib.load('model/input_columns.pkl')
output_columns = joblib.load('model/output_columns.pkl')

SUPPORTED_MODELS = {
    'random_forest': 'model/random_forest_model.pkl',
    'decision_tree': 'model/decision_tree_model.pkl',
    'linear_regression': 'model/linear_regression_model.pkl',
    'multivariate_linear_regression': 'model/multivariate_linear_regression_model.pkl',
    'xgboost': 'model/xgboost_model.pkl',
    'cnn': 'model/cnn_model.h5',
    'fnn': 'model/optimized_fnn_model.h5',
    'dnn': 'model/dnn_model.h5',
}

SCALERS = {
    'cnn': 'model/cnn_scaler.pkl',
    'fnn': 'model/optimized_fnn_scaler.pkl',
    'dnn': 'model/dnn_scaler.pkl',
}

@app.route('/')
def home():
    return "🚀 Thrust Pad Bearing ML Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("📩 Received Data:", data)

        model_name = data.get("model")
        if model_name not in SUPPORTED_MODELS:
            return jsonify({'error': f"Model '{model_name}' not supported."}), 400

        model_path = SUPPORTED_MODELS[model_name]
        is_deep = model_path.endswith('.h5')

        # Safely convert input data
        input_data = {k: float(v) for k, v in data.items() if k in input_columns}
        df = pd.DataFrame([input_data], columns=input_columns)

        if is_deep:
            # Load and apply scaler
            scaler = joblib.load(SCALERS[model_name])
            df_scaled = scaler.transform(df)

            # CNN needs reshaping
            if model_name == 'cnn':
                df_scaled = df_scaled.reshape(df_scaled.shape[0], df_scaled.shape[1], 1)

            # Load model without compiling
            model = load_model(model_path, compile=False)
            prediction = model.predict(df_scaled)
        else:
            model = joblib.load(model_path)
            prediction = model.predict(df)

        result = {k: float(v) for k, v in zip(output_columns, prediction[0])}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)