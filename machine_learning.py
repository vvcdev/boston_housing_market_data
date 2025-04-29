import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.express as px
import plotly.graph_objects as go

def load_data():
    """Load the Boston Housing dataset"""
    data = pd.read_csv('boston-housing-dataset.csv')
    # Remove unnamed index column if present
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    return data

def create_correlation_heatmap(data):
    """Create and return a correlation heatmap figure"""
    corr = data.corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Heatmap')
    return fig

def create_3d_correlation(data, x_col='RM', y_col='LSTAT', z_col='MEDV'):
    """Create a 3D scatter plot showing correlations"""
    fig = px.scatter_3d(
        data, 
        x=x_col, 
        y=y_col, 
        z=z_col,
        color='MEDV',
        opacity=0.7,
        title=f'3D Correlation between {x_col}, {y_col} and {z_col}'
    )
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        width=800,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def train_models(data, test_size=0.2, random_state=42):
    """Train ML models on the Boston Housing dataset"""
    # Prepare data
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    models = {
        'Linear Regression': lr_model,
        'Random Forest': rf_model
    }
    
    model_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        model_results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    # Save the models and scaler
    joblib.dump(models['Linear Regression'], 'lr_model.pkl')
    joblib.dump(models['Random Forest'], 'rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model_results, X_test, y_test, scaler

def feature_importance(model, feature_names):
    """Get feature importance for the Random Forest model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        features_ranked = [(feature_names[i], importances[i]) for i in indices]
        return features_ranked
    return None

def predict_sample(model, scaler, data, n_samples=50):
    """Predict housing prices for a random sample of data"""
    # Select random samples
    sample_indices = np.random.choice(data.shape[0], n_samples, replace=False)
    sample_data = data.iloc[sample_indices].copy()
    
    # Split features and target
    X_sample = sample_data.drop('MEDV', axis=1)
    y_true = sample_data['MEDV']
    
    # Scale features
    X_sample_scaled = scaler.transform(X_sample)
    
    # Make predictions
    y_pred = model.predict(X_sample_scaled)
    
    # Add predictions to the sample data
    sample_data['Predicted_MEDV'] = y_pred
    sample_data['Error'] = sample_data['Predicted_MEDV'] - sample_data['MEDV']
    sample_data['Abs_Error'] = abs(sample_data['Error'])
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mean_abs_error': sample_data['Abs_Error'].mean()
    }
    
    return sample_data, metrics

if __name__ == '__main__':
    # Test functionality
    data = load_data()
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Train models
    model_results, X_test, y_test, scaler = train_models(data)
    
    # Print evaluation metrics
    for name, results in model_results.items():
        print(f"\n{name} Results:")
        print(f"MSE: {results['mse']:.2f}")
        print(f"RMSE: {results['rmse']:.2f}")
        print(f"R²: {results['r2']:.2f}")
    
    # Feature importance for Random Forest
    rf_model = model_results['Random Forest']['model']
    feature_imp = feature_importance(rf_model, X_test.columns)
    
    if feature_imp:
        print("\nFeature Importance:")
        for feature, importance in feature_imp[:5]:
            print(f"{feature}: {importance:.4f}")
