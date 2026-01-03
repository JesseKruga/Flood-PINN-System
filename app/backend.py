from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import tensorflow as tf
import joblib
import os
import glob

# Import your utility functions
from model_utils import (
    FloodPredictionPINN, 
    physics_loss_function, 
    get_prediction_with_drivers
)

app = Flask(__name__)
CORS(app)

# Folder where your models are saved
MODELS_DIR = 'models' 

# --- GLOBAL DICTIONARY TO STORE LOADED MODELS ---
# Structure: { "Isoko South": { "model": object, "scaler": object }, ... }
loaded_artifacts = {}

def preload_artifacts():
    """
    Loads ALL models found in the models/ directory at startup.
    This prevents threading crashes that happen when loading inside a route.
    """
    print("="*50)
    print("🚀 PRE-LOADING ALL MODELS...")
    print("="*50)

    # Find all .keras model files
    model_files = glob.glob(os.path.join(MODELS_DIR, "*_model_test_run*.keras"))
    
    if not model_files:
        print(f"⚠️ No models found in {MODELS_DIR}. Please check file naming.")
        return

    custom_objects = {
        'PhysicsInformedModel': FloodPredictionPINN.PhysicsInformedModel,
        'physics_loss_function': physics_loss_function
    }

    for model_path in model_files:
        try:
            # 1. Infer LGA Name and Scaler Path from the Model Filename
            filename = os.path.basename(model_path)
            
            # Example filename: "Isoko_South_model_test_run.keras"
            # We want to extract: "Isoko South"
            base_name = filename.split('_model_test_run')[0] # "Isoko_South"
            lga_name = base_name.replace("_", " ") # "Isoko South"
            
            # Construct the matching scaler path
            # It should look like: "Isoko_South_scaler_test_run.pkl"
            scaler_filename = filename.replace("model", "scaler").replace(".keras", ".pkl")
            scaler_path = os.path.join(MODELS_DIR, scaler_filename)

            if not os.path.exists(scaler_path):
                print(f"⚠️ Scaler not found for {lga_name} (Expected: {scaler_filename}). Skipping.")
                continue

            # 2. Load them
            print(f"📂 Loading {lga_name}...")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            scaler = joblib.load(scaler_path)

            # 3. Store in Global Dictionary
            loaded_artifacts[lga_name] = {
                "model": model,
                "scaler": scaler
            }
            print(f"✅ {lga_name} Ready.")

        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
    
    print("="*50)
    print(f"🎉 System Ready. Loaded {len(loaded_artifacts)} LGAs: {list(loaded_artifacts.keys())}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_body = request.get_json()
        
        # 1. Extract Parameters
        raw_data = req_body.get('data')
        selected_date = req_body.get('selected_date')
        lga = req_body.get('lga') 

        if not raw_data or not selected_date or not lga:
            return jsonify({'error': 'Missing data, selected_date, or lga'}), 400

        # 2. Retrieve Pre-loaded Model
        # Normalize the name just in case (e.g. "Isoko South" vs "Isoko_South")
        lga_key = lga.replace("_", " ") # Ensure spaces
        
        artifact = loaded_artifacts.get(lga_key)
        
        if not artifact:
            # Try fuzzy match if exact match fails
            available = list(loaded_artifacts.keys())
            return jsonify({'error': f'Model for "{lga}" not loaded. Available: {available}'}), 404

        model = artifact['model']
        scaler = artifact['scaler']

        df = pd.DataFrame(raw_data)
        
        # 3. Run Prediction
        result = get_prediction_with_drivers(model, scaler, df, selected_date)

        if "error" in result:
            return jsonify(result), 400

        risk_labels = ['Low Risk', 'Moderate Risk', 'High Risk']
        
        response = {
            'lga': lga,
            'date': result['date'],
            'risk_level': risk_labels[result['risk_class']],
            'confidence': f"{result['confidence']:.2%}",
            'key_drivers': result['drivers']
        }

        return jsonify({'predictions': [response]})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/available_models', methods=['GET'])
def get_available_models():
    """Returns the list of LGAs that are currently loaded and ready."""
    return jsonify({'available_lgas': sorted(list(loaded_artifacts.keys()))})

if __name__ == '__main__':
    # LOAD EVERYTHING BEFORE SERVER STARTS
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    preload_artifacts()
    
    # Run in single-threaded mode to be extra safe with TF
    app.run(debug=True, port=5000, threaded=False)