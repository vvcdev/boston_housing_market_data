import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

# Import the machine learning functions
from machine_learning import (
    load_data, 
    create_correlation_heatmap, 
    create_3d_correlation,
    train_models, 
    feature_importance,
    predict_sample
)

# Set page configuration
st.set_page_config(
    page_title="Boston Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for different pages
def home_page():
    st.title("Boston Housing Price Predictor")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/20070701-Boston-CommonAndBeacon.jpg/1200px-20070701-Boston-CommonAndBeacon.jpg", width=800)
    
    st.write("""
    ## Welcome to the Boston Housing Price Prediction App!
    
    This application uses machine learning to predict housing prices in Boston based on various factors.
    
    ### Features in the Boston Housing Dataset:
    
    - **CRIM**: Per capita crime rate by town
    - **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft
    - **INDUS**: Proportion of non-retail business acres per town
    - **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    - **NOX**: Nitric oxides concentration (parts per 10 million)
    - **RM**: Average number of rooms per dwelling
    - **AGE**: Proportion of owner-occupied units built prior to 1940
    - **DIS**: Weighted distances to five Boston employment centers
    - **RAD**: Index of accessibility to radial highways
    - **TAX**: Full-value property-tax rate per $10,000
    - **PTRATIO**: Pupil-teacher ratio by town
    - **B**: 1000(Bk - 0.63)² where Bk is the proportion of Black people by town
    - **LSTAT**: % lower status of the population
    - **MEDV**: Median value of owner-occupied homes in $1000's (Target Variable)
    
    ### Use the sidebar to navigate through different sections of the application.
    """)

def data_exploration_page():
    st.title("Data Exploration")
    
    # Load the dataset
    data = load_data()
    
    st.write("### Boston Housing Dataset")
    st.write(f"Dataset shape: {data.shape}")
    
    # Display dataset
    st.write("#### Dataset Sample")
    st.dataframe(data.head(10))
    
    # Summary statistics
    st.write("#### Summary Statistics")
    st.dataframe(data.describe())
    
    # Display correlation heatmap
    st.write("### Correlation Analysis")
    st.write("#### Correlation Heatmap")
    fig = create_correlation_heatmap(data)
    st.pyplot(fig)
    
    # 3D correlation visualization
    st.write("### 3D Correlation Visualization")
    
    # Feature selection for 3D plot
    col1, col2, col3 = st.columns(3)
    with col1:
        x_feature = st.selectbox("Select X-axis feature", options=data.columns.tolist(), index=data.columns.get_loc("RM"))
    with col2:
        y_feature = st.selectbox("Select Y-axis feature", options=data.columns.tolist(), index=data.columns.get_loc("LSTAT"))
    with col3:
        z_feature = st.selectbox("Select Z-axis feature", options=data.columns.tolist(), index=data.columns.get_loc("MEDV"))
    
    # Create and display 3D scatter plot
    fig_3d = create_3d_correlation(data, x_feature, y_feature, z_feature)
    st.plotly_chart(fig_3d, use_container_width=True)

def model_training_page():
    st.title("Model Training")
    
    # Load the dataset
    data = load_data()
    
    # Train models button
    if st.button("Train Models"):
        with st.spinner("Training models... This may take a moment."):
            model_results, X_test, y_test, scaler = train_models(data)
            
            # Display results
            st.success("Models trained successfully!")
            
            # Display metrics in tabs
            model_tabs = st.tabs(list(model_results.keys()))
            
            for i, (name, results) in enumerate(model_results.items()):
                with model_tabs[i]:
                    st.write(f"### {name} Results:")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MSE", f"{results['mse']:.2f}")
                    col2.metric("RMSE", f"{results['rmse']:.2f}")
                    col3.metric("R²", f"{results['r2']:.2f}")
                    
                    # Feature importance for Random Forest
                    if name == "Random Forest":
                        st.write("### Feature Importance")
                        feature_imp = feature_importance(results['model'], X_test.columns)
                        
                        # Sort feature importance
                        feature_imp_df = pd.DataFrame(feature_imp, columns=['Feature', 'Importance'])
                        feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False)
                        
                        # Plot feature importance
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=feature_imp_df, ax=ax)
                        ax.set_title("Feature Importance")
                        st.pyplot(fig)
    else:
        st.info("Click the button above to train the models on the Boston Housing Dataset.")
        
        # Check if models already exist
        if os.path.exists('rf_model.pkl'):
            st.success("Models have been previously trained and are available for predictions.")

def prediction_page():
    st.title("Housing Price Prediction")
    
    # Check if models exist
    if not (os.path.exists('rf_model.pkl') and os.path.exists('lr_model.pkl') and os.path.exists('scaler.pkl')):
        st.error("Models not found. Please go to the Model Training page and train the models first.")
        return
    
    # Load models and scaler
    rf_model = joblib.load('rf_model.pkl')
    lr_model = joblib.load('lr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Load data to get column names
    data = load_data()
    features = data.drop('MEDV', axis=1).columns
    
    # Create input form
    st.write("### Enter Housing Details")
    
    # Create 3 columns
    col1, col2, col3 = st.columns(3)
    
    # Initialize input values dictionary
    input_values = {}
    
    # Column 1
    with col1:
        input_values['CRIM'] = st.number_input("Crime rate", 
                                               min_value=0.0, 
                                               max_value=100.0, 
                                               value=data['CRIM'].mean())
        input_values['ZN'] = st.number_input("Proportion of residential land zoned", 
                                             min_value=0.0, 
                                             max_value=100.0, 
                                             value=data['ZN'].mean())
        input_values['INDUS'] = st.number_input("Proportion of non-retail business acres", 
                                                min_value=0.0, 
                                                max_value=30.0, 
                                                value=data['INDUS'].mean())
        input_values['CHAS'] = st.selectbox("Charles River dummy variable", 
                                            options=[0, 1], 
                                            index=0)
        input_values['NOX'] = st.number_input("Nitric oxides concentration", 
                                              min_value=0.0, 
                                              max_value=1.0, 
                                              value=data['NOX'].mean())
    
    # Column 2
    with col2:
        input_values['RM'] = st.number_input("Average number of rooms", 
                                             min_value=3.0, 
                                             max_value=9.0, 
                                             value=data['RM'].mean())
        input_values['AGE'] = st.number_input("Proportion of owner-occupied units built prior to 1940", 
                                              min_value=0.0, 
                                              max_value=100.0, 
                                              value=data['AGE'].mean())
        input_values['DIS'] = st.number_input("Weighted distances to employment centers", 
                                              min_value=0.0, 
                                              max_value=15.0, 
                                              value=data['DIS'].mean())
        input_values['RAD'] = st.number_input("Index of accessibility to highways", 
                                              min_value=1.0, 
                                              max_value=24.0, 
                                              value=data['RAD'].mean())
        input_values['TAX'] = st.number_input("Property tax rate", 
                                              min_value=100.0, 
                                              max_value=1000.0, 
                                              value=data['TAX'].mean())
    
    # Column 3
    with col3:
        input_values['PTRATIO'] = st.number_input("Pupil-teacher ratio", 
                                                  min_value=10.0, 
                                                  max_value=25.0, 
                                                  value=data['PTRATIO'].mean())
        input_values['B'] = st.number_input("B variable", 
                                            min_value=0.0, 
                                            max_value=400.0, 
                                            value=data['B'].mean())
        input_values['LSTAT'] = st.number_input("% lower status of the population", 
                                                min_value=0.0, 
                                                max_value=40.0, 
                                                value=data['LSTAT'].mean())
    
    # Create input DataFrame
    input_df = pd.DataFrame([input_values])
    
    # Model selection
    model_choice = st.radio("Select Model for Prediction:", 
                            ["Linear Regression", "Random Forest"],
                            horizontal=True)
    
    if st.button("Predict Price"):
        # Scale the input data
        scaled_input = scaler.transform(input_df)
        
        # Make prediction based on the selected model
        if model_choice == "Linear Regression":
            prediction = lr_model.predict(scaled_input)[0]
            model_used = lr_model
        else:
            prediction = rf_model.predict(scaled_input)[0]
            model_used = rf_model
        
        # Display prediction
        st.success(f"### Predicted House Price: ${prediction*1000:.2f}")
        
        # Show factors influencing the prediction
        if model_choice == "Random Forest":
            st.write("### Feature Importance in This Prediction")
            
            # Get feature importances
            feature_imp = feature_importance(model_used, features)
            feature_imp_df = pd.DataFrame(feature_imp, columns=['Feature', 'Importance'])
            feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False)
            
            # Display top important features
            st.write("Top factors influencing the price prediction:")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(5), ax=ax)
            st.pyplot(fig)

def test_results_page():
    st.title("Model Testing with Random Sample")
    
    # Check if models exist
    if not (os.path.exists('rf_model.pkl') and os.path.exists('lr_model.pkl') and os.path.exists('scaler.pkl')):
        st.error("Models not found. Please go to the Model Training page and train the models first.")
        return
    
    # Load models and scaler
    rf_model = joblib.load('rf_model.pkl')
    lr_model = joblib.load('lr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Load data
    data = load_data()
    
    # Sample size and model selection
    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.slider("Sample Size", min_value=10, max_value=100, value=50)
    with col2:
        model_choice = st.radio("Select Model to Test:", 
                                ["Linear Regression", "Random Forest"],
                                horizontal=True)
    
    # Select model based on choice
    model = lr_model if model_choice == "Linear Regression" else rf_model
    
    # Test button
    if st.button("Test Model"):
        with st.spinner(f"Testing {model_choice} model on {sample_size} random samples..."):
            # Generate predictions for random samples
            sample_data, metrics = predict_sample(model, scaler, data, sample_size)
            
            # Display metrics
            st.success("#### Testing Complete!")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Squared Error", f"{metrics['mse']:.2f}")
            col2.metric("Root MSE", f"{metrics['rmse']:.2f}")
            col3.metric("R² Score", f"{metrics['r2']:.2f}")
            col4.metric("Mean Absolute Error", f"{metrics['mean_abs_error']:.2f}")
            
            # Visualization of actual vs predicted
            st.write("### Actual vs Predicted Values")
            
            # Scatter plot
            fig = px.scatter(
                sample_data, 
                x="MEDV", 
                y="Predicted_MEDV",
                hover_data=["CRIM", "RM", "LSTAT", "Error"],
                labels={"MEDV": "Actual Price ($1000s)", "Predicted_MEDV": "Predicted Price ($1000s)"}
            )
            
            # Add perfect prediction line
            min_val = min(sample_data["MEDV"].min(), sample_data["Predicted_MEDV"].min())
            max_val = max(sample_data["MEDV"].max(), sample_data["Predicted_MEDV"].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], 
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='green')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display error distribution
            st.write("### Error Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(sample_data["Error"], kde=True, ax=ax)
            ax.set_title("Distribution of Prediction Errors")
            ax.set_xlabel("Prediction Error")
            st.pyplot(fig)
            
            # Display sample data with predictions
            st.write("### Sample Data with Predictions")
            display_cols = ["CRIM", "RM", "LSTAT", "MEDV", "Predicted_MEDV", "Error"]
            st.dataframe(sample_data[display_cols], use_container_width=True)

# Main app structure
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
                           ["Home", "Data Exploration", "Model Training", "Make Prediction", "Test Results"])
    
    # Display the selected page
    if page == "Home":
        home_page()
    elif page == "Data Exploration":
        data_exploration_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Make Prediction":
        prediction_page()
    elif page == "Test Results":
        test_results_page()

if __name__ == "__main__":
    main()
