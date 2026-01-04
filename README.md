# Physics-Informed Flood Prediction System (PINN)

## 🌊 Overview
- This project implements a **Physics-Informed Neural Network (PINN)** to predict flood risks in Nigeria. Unlike standard deep learning models, this system integrates hydrological physical laws (via a ODE-based loss function) to constrain predictions, making it robust in data-scarce environments.
- To address the severe class imbalance (1:18) inherent in the flood training datasets, the system utilizes a **Regulated Time-variant SMOTE (T-SMOTE)** strategy.

To address the severe class imbalance (1:18) inherent in flood datasets, the system utilizes a **Regulated Time-variant SMOTE (T-SMOTE)** strategy.

## 🚀 Key Features
- **Physics-Informed Learning:** Incorporates a custom loss function derived from the water balance equation ($dS/dt = P - E - R$).
- **Regulated T-SMOTE:** Implements a custom class balancing strategy ($\alpha=0.4, k=3$) that restored Low-Risk precision to >95% while maintaining >70% Recall for critical flood events.
- **Spatial Generalization:** Validated on distinct geographical regions (Isoko South, Isoko North, Oshimili North) against their model weights (e.g Oshimilli data was used to validate Isoko South model to get predictions for reported flood occurences date).

## 📂 Project Structure
- `app/`: Deployment source code (Flask API & Streamlit UI).
- `notebooks/`: Research training pipeline.
- `plots/`: Diagnostic charts showing model convergence and confusion matrices.
- `data/`: **(Not included due to privacy restrictions)**.

## ⚡ How to Run
1. **Clone and Install:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Flood-PINN-System.git](https://github.com/YOUR_USERNAME/Flood-PINN-System.git)
   pip install -r requirements.txt
2. **To run the app**
- Run the backend using python app/backend.py
- Then, run the frontend using streamlit run app/frontend.py
- You can still validate the model using your data based on the input variables as seen in the frontend

##  📖 User Guide
To validate the model using the dashboard:
1.  **Select Region:** Choose the target Local Government Area (LGA) from the dropdown menu (e.g., Isoko South). This automatically loads the correct region-specific model weights.
2.  **Upload Data:** Upload your climatic CSV file (Max 200MB).
    * *Requirement:* The dataset must contain at least 30 days of continuous data prior to your target date.
3.  **Map Columns:** Use the interface to map your CSV column names to the required model inputs (Precipitation, Soil Moisture, etc.).
4.  **Select Target Date:** Choose the specific date you wish to analyze.
5.  **Generate:** Click "Generate Predictions" to view the Risk Level, Confidence Score, and Key Physical Drivers.

> **Tip for Forecasting:** To predict future flood risks, upload a CSV containing both historical data and forecasted weather data. Then, select a future date within the forecasted range as your target.
