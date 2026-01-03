import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px

# Constants
FLASK_URL = "http://127.0.0.1:5000/predict"
AVAILABLE_MODELS_URL = "http://127.0.0.1:5000/available_models" # Endpoint to fetch LGAs

REQUIRED_INPUTS = [
    'Ppt', 'Evapotranspiration', 'Runoff', 'SoilMoisture',
    'MinTemp', 'MaxTemp', 'HargreavesEvapotranspiration',
    'WindSpeed', 'VaporPressure', 'SoilSurfaceMoisture'
]

st.set_page_config(page_title="Flood Risk PINN System", layout="wide")

st.title("🌊 Physics-Informed Flood Prediction System")
st.markdown("""
This system uses a **PINN (Physics-Informed Neural Network)** to predict flood risk.
**Note:** To predict a specific date, your uploaded CSV must contain data (historical or forecasted) for the preceding 30 days.
""")

# --- Step 0: Select LGA (Dynamic Fetching) ---
st.subheader("📍 Select Region")
try:
    # Fetch available LGAs from backend to ensure we only show valid options
    response = requests.get(AVAILABLE_MODELS_URL)
    if response.status_code == 200:
        lga_options = response.json().get('available_lgas', [])
    else:
        lga_options = ["Isoko South", "Isoko North", "Oshimili North"] # Fallback
except:
    lga_options = ["Isoko South", "Isoko North", "Oshimili North"] # Fallback if backend is down

selected_lga = st.selectbox("Select Local Government Area (LGA)", options=lga_options)


# --- Step 1: File Upload ---
uploaded_file = st.file_uploader("Upload Climatic Data (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Ensure date column is datetime for sorting/filtering in UI
    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    # --- Step 2: Column Mapping ---
    st.subheader("🛠️ Map Your Columns")
    col1, col2 = st.columns(2)
    column_mapping = {}
    
    # Date Mapping
    with col1:
        date_col = st.selectbox("Select Date Column", options=df.columns)
        column_mapping['Date'] = date_col

    # Feature Mapping
    all_mapped = True
    for i, req_col in enumerate(REQUIRED_INPUTS):
        target_col = col1 if i % 2 == 0 else col2
        
        # Auto-find column
        default_index = 0
        for idx, col_name in enumerate(df.columns):
            if req_col.lower() in col_name.lower():
                default_index = idx
                break
        
        with target_col:
            user_col = st.selectbox(f"Map for **{req_col}**", options=['Select...'] + list(df.columns), index=default_index+1 if default_index else 0)
            if user_col == 'Select...':
                all_mapped = False
            else:
                column_mapping[req_col] = user_col

    # --- Step 3: Select Prediction Date ---
    if all_mapped:
        st.divider()
        st.subheader("📅 Select Prediction Target")
        
        # Convert user's date column to datetime objects for the date picker
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            min_date = df[date_col].min().date()
            max_date = df[date_col].max().date()
            
            selected_date = st.date_input(
                "Choose a date to analyze:",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        except Exception as e:
            st.error(f"Error parsing date column: {e}")
            selected_date = None

        if st.button("Generate Prediction"):
            if not selected_lga:
                 st.error("Please select an LGA.")
            else:
                with st.spinner(f"Analyzing flood risk for {selected_lga} on {selected_date}..."):
                    try:
                        # 1. Prepare Dataframe (Rename & Format)
                        processed_df = df.rename(columns={v: k for k, v in column_mapping.items() if k != 'Date'})
                        processed_df['Date'] = df[date_col].dt.strftime('%Y-%m-%d') # Standardize date format
                        
                        # Filter only needed columns
                        cols_to_send = ['Date'] + REQUIRED_INPUTS
                        payload_df = processed_df[cols_to_send]

                        # 2. Construct Payload (Added 'lga')
                        payload = {
                            "lga": selected_lga,
                            "selected_date": str(selected_date),
                            "data": payload_df.to_dict(orient='records')
                        }

                        # 3. Send Request
                        response = requests.post(FLASK_URL, json=payload)
                        
                        if response.status_code == 200:
                            # The backend now returns a list with 1 result object
                            data = response.json()['predictions'][0]
                            
                            # --- Display Results ---
                            st.success("Analysis Complete")
                            
                            # Metrics
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Date", data['date'])
                            m1.metric("Region", data['lga']) # Show Region
                            m1.metric("Risk Level", data['risk_level'], 
                                    delta_color="inverse" if data['risk_level'] == "High Risk" else "normal")
                            m2.metric("Confidence", data['confidence'])
                            
                            # Drivers
                            m3.write("**Key Drivers (Physical Factors):**")
                            for driver in data['key_drivers']:
                                m3.info(driver) # Uses blue info box for cleaner look

                        else:
                            st.error(f"Backend Error: {response.json().get('error')}")

                    except Exception as e:
                        st.error(f"Connection Error: {e}. Ensure Flask is running on port 5000.")