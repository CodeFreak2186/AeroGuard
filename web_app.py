"""
Professional Web Dashboard for Jet Engine Predictive Maintenance
=================================================================
A modern, responsive Flask web application with professional UI/UX
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.dirname(__file__))

# Import modules
try:
    from train_balanced import BalancedFeatureEngineer
except ImportError:
    pass

from health_score import rul_to_health_percentage, get_health_status, get_maintenance_recommendation
from data_loader import CMAPSSDataLoader

app = Flask(__name__)

# Global variables for model and data
model = None
scaler = None
test_data = None
predictions = None
data_loader = None

def load_models():
    """Load the trained model and scaler"""
    global model, scaler
    
    model_path = os.path.join('models', 'balanced_xgboost_model.pkl')
    scaler_path = os.path.join('models', 'balanced_scaler.pkl')
    
    if not os.path.exists(model_path):
        model_path = os.path.join('models', 'xgboost_model.pkl')
        scaler_path = os.path.join('models', 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"✓ Loaded model from {model_path}")

def load_and_predict():
    """Load test data and make predictions"""
    global test_data, predictions, data_loader
    
    # Initialize data loader and load test data
    data_loader = CMAPSSDataLoader()
    df_test = data_loader.load_raw_data('data/test_FD001.txt')
    
    # Check if we're using the balanced model (with feature engineering)
    model_path = os.path.join('models', 'balanced_xgboost_model.pkl')
    if os.path.exists(model_path):
        # Apply feature engineering for balanced model
        print("Applying feature engineering for balanced model...")
        from train_balanced import BalancedFeatureEngineer
        feature_engineer = BalancedFeatureEngineer()
        df_test = feature_engineer.add_balanced_features(df_test)
    
    # Load the scaler (should match the one used during training)
    scaler_path = os.path.join('models', 'balanced_scaler.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join('models', 'scaler.pkl')
    data_loader.load_scaler(scaler_path)
    
    # Prepare features (without RUL for test data)
    X_test, _ = data_loader.prepare_features(df_test, is_training=False)
    
    # Make predictions
    predictions = model.predict(X_test.values)
    
    # Combine with original data for display (use original df_test for basic info)
    test_data = data_loader.load_raw_data('data/test_FD001.txt')
    test_data['predicted_rul'] = predictions
    test_data['health_percentage'] = test_data['predicted_rul'].apply(rul_to_health_percentage)
    # Store just the status text, not the tuple
    test_data['health_status'] = test_data['health_percentage'].apply(lambda x: get_health_status(x)[0])
    
    # Rename engine_id to unit_id for consistency
    if 'engine_id' in test_data.columns:
        test_data.rename(columns={'engine_id': 'unit_id'}, inplace=True)
    
    print(f"✓ Predictions generated for {len(test_data)} samples")

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/overview')
def get_overview():
    """Get overview statistics"""
    last_cycle_data = test_data.groupby('unit_id').last()
    
    # Get status properly - health_status contains tuple, extract first element
    statuses = last_cycle_data['health_percentage'].apply(lambda x: get_health_status(x)[0])
    critical_count = sum(statuses == 'Critical')
    warning_count = sum(statuses == 'Warning')
    healthy_count = sum(statuses == 'Healthy')
    
    avg_health = last_cycle_data['health_percentage'].mean()
    avg_rul = last_cycle_data['predicted_rul'].mean()
    
    return jsonify({
        'total_engines': len(last_cycle_data),
        'critical': int(critical_count),
        'warning': int(warning_count),
        'healthy': int(healthy_count),
        'avg_health': float(avg_health),
        'avg_rul': float(avg_rul),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/engines')
def get_engines():
    """Get list of all engines with their current status"""
    last_cycle_data = test_data.groupby('unit_id').last().reset_index()
    
    engines = []
    for _, row in last_cycle_data.iterrows():
        status_text, _, _ = get_health_status(row['health_percentage'])
        engines.append({
            'unit_id': int(row['unit_id']),
            'health_percentage': float(row['health_percentage']),
            'predicted_rul': float(row['predicted_rul']),
            'status': status_text,
            'recommendation': get_maintenance_recommendation(row['health_percentage'], row['predicted_rul'])
        })
    
    # Sort by health percentage (critical first)
    engines.sort(key=lambda x: x['health_percentage'])
    
    return jsonify(engines)

@app.route('/api/engine/<int:engine_id>')
def get_engine_details(engine_id):
    """Get detailed information for a specific engine"""
    engine_data = test_data[test_data['unit_id'] == engine_id].copy()
    
    if engine_data.empty:
        return jsonify({'error': 'Engine not found'}), 404
    
    # Get time series data
    cycles = engine_data['cycle'].tolist()
    rul_values = engine_data['predicted_rul'].tolist()
    health_values = engine_data['health_percentage'].tolist()
    
    # Get sensor data for the last cycle
    last_cycle = engine_data.iloc[-1]
    sensor_cols = [col for col in engine_data.columns if col.startswith('sensor_')]
    sensor_data = {col: float(last_cycle[col]) for col in sensor_cols}
    
    # Calculate sensor statistics
    sensor_stats = {}
    for col in sensor_cols:
        sensor_stats[col] = {
            'current': float(last_cycle[col]),
            'mean': float(engine_data[col].mean()),
            'std': float(engine_data[col].std()),
            'min': float(engine_data[col].min()),
            'max': float(engine_data[col].max())
        }
    
    # Get status properly
    status_text, _, _ = get_health_status(last_cycle['health_percentage'])
    
    return jsonify({
        'unit_id': int(engine_id),
        'current_cycle': int(last_cycle['cycle']),
        'predicted_rul': float(last_cycle['predicted_rul']),
        'rmse': 5.5, # Hardcoded based on advanced model performance
        'health_percentage': float(last_cycle['health_percentage']),
        'status': status_text,
        'recommendation': get_maintenance_recommendation(last_cycle['health_percentage'], last_cycle['predicted_rul']),
        'cycles': cycles,
        'rul_history': rul_values,
        'health_history': health_values,
        'sensors': sensor_data,
        'sensor_stats': sensor_stats
    })

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data for charts"""
    last_cycle_data = test_data.groupby('unit_id').last()
    
    # Get status properly
    statuses = last_cycle_data['health_percentage'].apply(lambda x: get_health_status(x)[0])
    
    # Health distribution - FIXED
    health_distribution = {
        'Critical': int(sum(statuses == 'Critical')),
        'Warning': int(sum(statuses == 'Warning')),
        'Healthy': int(sum(statuses == 'Healthy'))
    }
    
    # RUL distribution bins
    rul_bins = [0, 50, 100, 150, 200, 250, 300]
    rul_hist, _ = np.histogram(last_cycle_data['predicted_rul'], bins=rul_bins)
    rul_distribution = {
        f"{rul_bins[i]}-{rul_bins[i+1]}": int(rul_hist[i]) 
        for i in range(len(rul_hist))
    }
    
    # Health percentage distribution
    health_bins = [0, 20, 40, 60, 80, 100]
    health_hist, _ = np.histogram(last_cycle_data['health_percentage'], bins=health_bins)
    health_percentage_dist = {
        f"{health_bins[i]}-{health_bins[i+1]}%": int(health_hist[i]) 
        for i in range(len(health_hist))
    }
    
    # Get top sensors with highest variation
    sensor_cols = [col for col in test_data.columns if col.startswith('sensor_')]
    sensor_importance = {}
    for col in sensor_cols[:10]:  # Top 10 sensors
        sensor_importance[col.replace('sensor_', 'Sensor ')] = float(
            test_data[col].std() / (test_data[col].mean() + 1e-10)
        )
    
    # Fleet Trends (Avg Health & RUL vs Cycle)
    trend_data = test_data.groupby('cycle')[['health_percentage', 'predicted_rul']].mean().reset_index()
    # Limit to reasonable cycle count for visualization
    trend_data = trend_data[trend_data['cycle'] <= 200]
    
    fleet_trend = {
        'cycles': trend_data['cycle'].tolist(),
        'avg_health': [float(x) for x in trend_data['health_percentage'].tolist()],
        'avg_rul': [float(x) for x in trend_data['predicted_rul'].tolist()]
    }

    # Model Comparison Data (from BREAKTHROUGH_RESULTS.md)
    model_comparison = {
        'metrics': ['R² Score', 'RMSE (cycles)', 'Variance Explained (%)'],
        'baseline': [0.85, 16.0, 85.0],
        'advanced': [0.99, 5.5, 99.5]
    }
    
    return jsonify({
        'health_distribution': health_distribution,
        'rul_distribution': rul_distribution,
        'health_percentage_distribution': health_percentage_dist,
        'sensor_importance': dict(sorted(sensor_importance.items(), key=lambda x: x[1], reverse=True)[:8]),
        'fleet_trend': fleet_trend,
        'model_comparison': model_comparison
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Loading Jet Engine Predictive Maintenance Dashboard")
    print("=" * 60)
    
    # Load models and data
    load_models()
    load_and_predict()
    
    print("=" * 60)
    print("✓ Dashboard ready!")
    print("🌐 Open your browser and visit: http://localhost:5000")
    print("=" * 60)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
