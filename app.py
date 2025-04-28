import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    .stDataFrame {
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Define paths for saving models
model_path = "./saved_models/air_quality_random_forest_model.joblib"
feature_names_path = "./saved_models/air_quality_feature_names.joblib"
os.makedirs("./saved_models", exist_ok=True)  # Create directory if it doesn't exist

# Load or Train Model
def load_or_train_model(data):
    force_retrain = st.sidebar.checkbox("Force Retrain Model", value=False)
    
    if os.path.exists(model_path) and os.path.exists(feature_names_path) and not force_retrain:
        st.sidebar.success("Loading pre-trained model and feature names...")
        model = joblib.load(model_path)
        feature_names = joblib.load(feature_names_path)
        return model, feature_names, None, None
    
    st.sidebar.warning("No pre-trained model found. Training a new model...")
    
    # Feature Engineering
    data['AQI_lag1'] = data['AQI'].shift(1)
    data.dropna(inplace=True)

    # Define features and target
    X = data.drop(['AQI', 'Date', 'City', 'AQI_Bucket'], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    y = data['AQI']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with parallel processing
    with st.spinner("Training model (this may take a while)..."):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  # Use all CPU cores
        model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    st.sidebar.success(f"Model trained successfully! RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # Save model and feature names
    joblib.dump(model, model_path)
    joblib.dump(X_train.columns.tolist(), feature_names_path)
    feature_names = X_train.columns.tolist()
    
    return model, feature_names, y_test, y_pred

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv('combined_daily_data.csv', dtype={'AQI_Bucket': 'str'})
    data['Date'] = pd.to_datetime(data['Date'])
    data.ffill(inplace=True)
    data.dropna(subset=['AQI'], inplace=True)
    return data

# Main App
def main():
    st.title("🌍 Air Quality Prediction Dashboard")
    
    # Load Data
    data = load_data()

    # Sidebar for Navigation
    app_mode = st.sidebar.radio("Choose a section", ["EDA", "Predict AQI"])

    if app_mode == "EDA":
        st.header("Exploratory Data Analysis (EDA)")
        
        # Display Dataset
        if st.checkbox("Show Raw Data"):
            st.subheader("Raw Data")
            st.write(data)

        # AQI Distribution Plot
        st.subheader("AQI Distribution")
        fig = px.histogram(data, x='AQI', nbins=50, color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig, use_container_width=True)

        # Time Series Plot
        st.subheader("AQI Over Time")
        time_series_data = data.groupby('Date')['AQI'].mean().reset_index()
        fig = px.line(time_series_data, x='Date', y='AQI', title="AQI Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # Box Plot by City
        st.subheader("AQI Distribution by City")
        fig = px.box(data, x='City', y='AQI', color='City', title="AQI Distribution by City")
        st.plotly_chart(fig, use_container_width=True)

        # Scatter Plot: PM2.5 vs AQI
        st.subheader("PM2.5 vs AQI")
        fig = px.scatter(data, x='PM2.5', y='AQI', trendline="ols", title="PM2.5 vs AQI")
        st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        corr = numeric_data.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Blues'
        ))
        st.plotly_chart(fig, use_container_width=True)

    elif app_mode == "Predict AQI":
        st.header("Predict Air Quality Index (AQI)")
        
        # Load or Train Model and Feature Names
        model, feature_names, y_test, y_pred = load_or_train_model(data)

        # Input Form for Prediction Features
        col1, col2 = st.columns(2)

        with col1:
            pm25 = st.number_input("PM2.5", value=0.0)
            pm10 = st.number_input("PM10", value=0.0)
            no = st.number_input("NO", value=0.0)
            no2 = st.number_input("NO2", value=0.0)
        with col2:
            nox = st.number_input("NOx", value=0.0)
            nh3 = st.number_input("NH3", value=0.0)
            co = st.number_input("CO", value=0.0)
            so2 = st.number_input("SO2", value=0.0)

        aqi_lag1 = st.number_input("Previous Day's AQI (Lag1)", min_value=0.0)

        # Predict Button
        if st.button("Predict AQI"):
            input_data = pd.DataFrame({
                'PM2.5': [pm25],
                'PM10': [pm10],
                'NO': [no],
                'NO2': [no2],
                'NOx': [nox],
                'NH3': [nh3],
                'CO': [co],
                'SO2': [so2],
                'AQI_lag1': [aqi_lag1]
            })

            # Reindex to ensure we have the same feature set as the training data
            input_data = input_data.reindex(columns=feature_names, fill_value=0)

            prediction = model.predict(input_data)[0]
            st.success(f"Predicted AQI: **{prediction:.2f}**")

            # AQI Bucket Insights
            if prediction <= 50:
                st.info("Good: Air quality is satisfactory, and air pollution poses little or no risk.")
            elif prediction <= 100:
                st.info("Moderate: Air quality is acceptable; however, there may be a risk for some people.")
            elif prediction <= 150:
                st.warning("Unhealthy for Sensitive Groups: Members of sensitive groups may experience health effects.")
            elif prediction <= 200:
                st.error("Unhealthy: Everyone may begin to experience health effects.")
            elif prediction <= 300:
                st.error("Very Unhealthy: Health alert; everyone may experience more serious health effects.")
            else:
                st.error("Hazardous: Health warning of emergency conditions; the entire population is affected.")

        # Display Evaluation Metrics and Plots
        if y_test is not None and y_pred is not None:
            st.subheader("Model Evaluation Metrics")
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

            st.subheader("Actual vs Predicted AQI")
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual AQI', 'y': 'Predicted AQI'}, title="Actual vs Predicted AQI")
            fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', name='Ideal Fit'))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Residual Plot")
            residuals = y_test - y_pred
            fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted AQI', 'y': 'Residuals'}, title="Residual Plot")
            fig.add_trace(go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0], mode='lines', name='Zero Residual Line'))
            st.plotly_chart(fig, use_container_width=True)

# Run the App
if __name__ == "__main__":
    main()