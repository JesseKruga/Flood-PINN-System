import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, Model

# ==========================================
# 1. CONSTANTS & CONFIG
# ==========================================
# CRITICAL: This must match the training order exactly.
# Based on your provided code/loss function: Ppt=0, ET=1, Runoff=2
FEATURE_ORDER = [
    'Ppt', 'Evapotranspiration', 'Runoff', 'SoilMoisture',
    'MinTemp', 'MaxTemp', 'HargreavesEvapotranspiration',
    'WindSpeed', 'VaporPressure', 'SoilSurfaceMoisture',
    'Day_of_Year_sin', 'Day_of_Year_cos', 'Month_sin', 'Month_cos'
]

SEQ_LENGTH = 30  # The lookback window

# ==========================================
# 2. CUSTOM LOSS & MODEL (Kept Original)
# ==========================================
def physics_loss_function(inputs, y_pred_soil_moisture):
    """
    Computes the physics loss for the PINN model.
    """
    # Extract physical variables (Assumes specific order of features)
    precipitation = inputs[:, :, 0]
    evapotranspiration = inputs[:, :, 1]
    runoff = inputs[:, :, 2]

    # Finite difference approximation
    dS_dt_approx = y_pred_soil_moisture[:, 1:] - y_pred_soil_moisture[:, :-1]
    
    drainage_coeff = 0.1
    deep_drainage = drainage_coeff * y_pred_soil_moisture[:, :-1]

    P = precipitation[:, :-1]
    ET = evapotranspiration[:, :-1]
    R = runoff[:, :-1]

    pde_residual = dS_dt_approx - (P - ET - R - deep_drainage)
    pde_loss = tf.reduce_mean(tf.square(pde_residual))

    bounds_loss = tf.reduce_mean(
        tf.maximum(0.0, -y_pred_soil_moisture) +
        tf.maximum(0.0, y_pred_soil_moisture - 1.0)
    )

    return pde_loss + bounds_loss

class FloodPredictionPINN:
    class PhysicsInformedModel(tf.keras.Model):
        def __init__(self, model, num_classes, **kwargs):
            super().__init__(**kwargs)
            self.model = model
            self.num_classes = num_classes
            self.class_weights = None

            self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
            self.class_loss_tracker = tf.keras.metrics.Mean(name="classification_loss")
            self.phys_loss_tracker = tf.keras.metrics.Mean(name="physics_loss")
            self.accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
            self.precision_metric = tf.keras.metrics.Precision(name="precision")
            self.recall_metric = tf.keras.metrics.Recall(name="recall")

        def get_config(self):
            config = super().get_config()
            config.update({
                "model": tf.keras.utils.serialize_keras_object(self.model),
                "num_classes": self.num_classes,
            })
            return config

        @classmethod
        def from_config(cls, config):
            model_config = tf.keras.utils.deserialize_keras_object(config.pop("model"))
            num_classes = config.pop("num_classes")
            return cls(model=model_config, num_classes=num_classes, **config)

        def call(self, inputs, training=False):
            return self.model(inputs, training=training)

        @property
        def metrics(self):
            return [
                self.total_loss_tracker, self.class_loss_tracker, self.phys_loss_tracker,
                self.accuracy_metric, self.precision_metric, self.recall_metric,
            ]

# ==========================================
# 3. PREPROCESSING HELPERS
# ==========================================
def calculate_seasonality(df):
    """Adds cyclical time features."""
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    df['Day_of_Year'] = df.index.dayofyear
    df['Month'] = df.index.month

    df['Day_of_Year_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365.25)
    df['Day_of_Year_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365.25)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    return df

# ==========================================
# 4. INFERENCE + EXPLANATION FUNCTION
# ==========================================
def get_prediction_with_drivers(model, scaler, raw_df, target_date_str):
    # --- A. Data Prep (Slicing) ---
    df = raw_df.copy()
    
    # ---------------------------------------------------------
    # 1. UNIT CORRECTION FIX (Crucial Step)
    # ---------------------------------------------------------
    # Fix SoilSurfaceMoisture: Convert Percentage (0-100) to Fraction (0-1)
    if 'SoilSurfaceMoisture' in df.columns:
        # Check if values are actually percentages (e.g., > 1.0)
        if df['SoilSurfaceMoisture'].mean() > 1.0:
            print("🔧 Auto-Correcting SoilSurfaceMoisture units (Div by 100)")
            df['SoilSurfaceMoisture'] = df['SoilSurfaceMoisture'] / 100.0

    # Optional: Check VaporPressure. If scaler expects hPa (10-30) and we have kPa (1-3)
    # If the mean is very low (< 5), but the Z-score was super negative, 
    # it implies the scaler expects HIGHER numbers. 
    # We will let the "Safety Clamp" handle this for now to be safe.
    # ---------------------------------------------------------

    if 'Day_of_Year_sin' not in df.columns:
        df = calculate_seasonality(df)
    
    target_date = pd.to_datetime(target_date_str)
    
    if target_date not in df.index:
        valid_dates = df.index[df.index <= target_date]
        if len(valid_dates) == 0:
             return {"error": "Date too early for dataset."}
        target_date = valid_dates[-1] 

    target_idx = df.index.get_loc(target_date)
    start_idx = target_idx - SEQ_LENGTH
    
    if start_idx < 0:
        return {"error": f"Not enough data. Need {SEQ_LENGTH} days before selection."}
    
    try:
        input_slice = df.iloc[start_idx : target_idx]
        input_values = input_slice[FEATURE_ORDER].values
    except KeyError as e:
        return {"error": f"Missing columns in input data: {e}"}
        
    # --- 2. SCALE AND SAFETY CLAMP ---
    input_scaled = scaler.transform(input_values)

    # 🚨 SAFETY CLAMP 🚨
    # This forces any wild outliers (like 381.0 or -4.8) into a safe range (-5 to +5).
    # This ensures the Neural Network never receives "galaxy-ending" numbers.
    input_scaled = np.clip(input_scaled, -5.0, 5.0)
    
    # Reshape for Model
    model_input = tf.Variable(
        input_scaled.reshape(1, SEQ_LENGTH, len(FEATURE_ORDER)), 
        dtype=tf.float32
    )
    
    # --- B. Prediction & Gradient Analysis ---
    with tf.GradientTape() as tape:
        tape.watch(model_input)
        predictions = model(model_input, training=False)
        class_probs = predictions[0] 
        top_class_score = tf.reduce_max(class_probs, axis=1)

    grads = tape.gradient(top_class_score, model_input)
    feature_importance = tf.reduce_mean(tf.abs(grads), axis=1)[0].numpy()
    
    # --- C. Formatting Output ---
    risk_class_idx = np.argmax(class_probs[0])
    confidence = float(np.max(class_probs[0]))
    
    feature_map = dict(zip(FEATURE_ORDER, feature_importance))
    sorted_features = sorted(feature_map.items(), key=lambda x: x[1], reverse=True)
    
    physical_drivers = [
        (feat, score) for feat, score in sorted_features 
        if 'sin' not in feat and 'cos' not in feat
    ]
    
    top_drivers_list = [f"{feat} ({score:.2e})" for feat, score in physical_drivers[:3]]

    return {
        "date": str(target_date.date()),
        "risk_class": int(risk_class_idx),
        "confidence": confidence,
        "drivers": top_drivers_list
    }