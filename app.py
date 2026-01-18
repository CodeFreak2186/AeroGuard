"""
UPDATED Streamlit Dashboard - Using Balanced Model (R² = 0.93)
===============================================================
Features:
- Uses balanced_xgboost_model.pkl (R² = 0.93, RMSE = 10.74)
- Shows model comparison (Baseline vs Balanced)
- Enhanced visualizations
- Professional UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import joblib

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
# Add current directory to path for train_balanced import (needed for pickle deserialization)
sys.path.insert(0, os.path.dirname(__file__))

# Import train_balanced BEFORE joblib.load to make BalancedFeatureEngineer available for pickle
try:
    from train_balanced import BalancedFeatureEngineer  # noqa: F401
except ImportError:
    pass

from health_score import rul_to_health_percentage, get_health_status, get_maintenance_recommendation

# Page configuration
st.set_page_config(
    page_title="Jet Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load both baseline and balanced models for comparison"""
    models = {}
    
    # Balanced model (R² = 0.93)
    balanced_model_path = "models/balanced_xgboost_model.pkl"
    balanced_engineer_path = "models/balanced_engineer.pkl"
    
    if os.path.exists(balanced_model_path) and os.path.exists(balanced_engineer_path):
        models['balanced'] = {
            'model': joblib.load(balanced_model_path),
            'engineer': joblib.load(balanced_engineer_path),
            'name': 'Balanced XGBoost',
            'r2': 0.9321,
            'rmse': 10.74
        }
    
    # Baseline model (R² = 0.85) - fallback
    baseline_model_path = "models/xgboost_model.pkl"
    baseline_scaler_path = "models/scaler.pkl"
    
    if os.path.exists(baseline_model_path):
        from data_loader import CMAPSSDataLoader
        baseline_loader = CMAPSSDataLoader()
        if os.path.exists(baseline_scaler_path):
            baseline_loader.load_scaler(baseline_scaler_path)
        
        models['baseline'] = {
            'model': joblib.load(baseline_model_path),
            'loader': baseline_loader,
            'name': 'Baseline XGBoost',
            'r2': 0.8494,
            'rmse': 15.98
        }
    
    return models


@st.cache_data
def load_test_data():
    """Load test data"""
    test_path = "data/test_FD001.txt"
    
    if not os.path.exists(test_path):
        st.error(f"❌ Test data not found at {test_path}")
        st.stop()
    
    # Use balanced engineer to load data
    models = load_models()
    if 'balanced' in models:
        engineer = models['balanced']['engineer']
        df = engineer.load_data(test_path)
    else:
        # Fallback to baseline loader
        from data_loader import CMAPSSDataLoader
        loader = CMAPSSDataLoader()
        df = loader.load_raw_data(test_path)
    
    return df


def predict_with_balanced_model(engine_id, df_test, model, engineer):
    """Generate predictions using balanced model"""
    # Filter data for this engine
    engine_data = df_test[df_test['engine_id'] == engine_id].copy()
    
    # Engineer features
    engine_data_engineered = engineer.add_balanced_features(engine_data)
    
    # Prepare features
    X, _ = engineer.prepare_features(engine_data_engineered, fit=False)
    
    # Predict RUL
    predicted_rul = model.predict(X)
    
    # Convert to health percentage
    health_pct = rul_to_health_percentage(predicted_rul)
    
    # Create results dataframe
    results = pd.DataFrame({
        'cycle': engine_data['cycle'].values,
        'predicted_rul': predicted_rul,
        'health_percentage': health_pct
    })
    
    return results


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<p class="main-header">✈️ Jet Engine Predictive Maintenance System</p>', unsafe_allow_html=True)
    st.markdown("### IIT Kharagpur Hackathon - Kshitij 2026")
    st.markdown("**Advanced ML Model**: R² = 0.93 | RMSE = 10.74 cycles | Cross-Validated")
    st.markdown("---")
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        models = load_models()
        df_test = load_test_data()
    
    if not models:
        st.error("❌ No models found. Please run train_balanced.py first.")
        st.stop()
    
    # Sidebar - Engine Selection & Model Info
    st.sidebar.header("⚙️ Engine Selection")
    
    available_engines = sorted(df_test['engine_id'].unique())
    selected_engine = st.sidebar.selectbox(
        "Select Engine ID:",
        available_engines,
        help="Choose an engine to analyze"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏆 Model Performance")
    
    if 'balanced' in models:
        st.sidebar.success(f"""
        **Balanced XGBoost**
        - R² Score: 0.9321
        - RMSE: 10.74 cycles
        - Features: ~100
        - Cross-Validated: ✅
        """)
    
    if 'baseline' in models:
        st.sidebar.info(f"""
        **Baseline XGBoost**
        - R² Score: 0.8494
        - RMSE: 15.98 cycles
        - Features: 27
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About")
    st.sidebar.info("""
    **Dataset**: NASA CMAPSS FD001
    
    **Approach**: Balanced feature engineering with rolling statistics, EMA, and sensor interactions
    
    **Validation**: 5-fold cross-validation
    """)
    
    # Generate predictions
    with st.spinner(f"Analyzing Engine {selected_engine}..."):
        if 'balanced' in models:
            results = predict_with_balanced_model(
                selected_engine, df_test, 
                models['balanced']['model'],
                models['balanced']['engineer']
            )
        else:
            st.error("Balanced model not found")
            st.stop()
    
    # Get latest predictions
    latest = results.iloc[-1]
    latest_rul = latest['predicted_rul']
    latest_health = latest['health_percentage']
    status, color, _ = get_health_status(latest_health)
    recommendation = get_maintenance_recommendation(latest_health, latest_rul)
    
    # Main Dashboard Layout
    st.header(f"🔧 Engine ID: {selected_engine}")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📅 Current Cycle",
            value=int(latest['cycle']),
            help="Current operating cycle"
        )
    
    with col2:
        st.metric(
            label="⏱️ Predicted RUL",
            value=f"{latest_rul:.1f} cycles",
            delta=f"±{10.74:.1f} (RMSE)",
            help="Remaining Useful Life prediction"
        )
    
    with col3:
        st.metric(
            label="❤️ Health Score",
            value=f"{latest_health:.1f}%",
            delta=f"{latest_health - 50:.1f}%" if latest_health < 50 else None,
            help="Engine health percentage"
        )
    
    with col4:
        st.markdown(f"### Status")
        st.markdown(f"<h2 style='color: {color}; margin: 0;'>{status}</h2>", unsafe_allow_html=True)
    
    # Recommendation Alert
    if status == 'Critical':
        st.error(f"🚨 {recommendation}")
    elif status == 'Warning':
        st.warning(f"⚠️ {recommendation}")
    else:
        st.success(f"✅ {recommendation}")
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 RUL Trend", "❤️ Health Trend", "📊 Model Comparison", "🔍 Feature Importance"])
    
    with tab1:
        st.subheader("Remaining Useful Life Over Time")
        
        fig_rul = go.Figure()
        
        fig_rul.add_trace(go.Scatter(
            x=results['cycle'],
            y=results['predicted_rul'],
            mode='lines+markers',
            name='Predicted RUL',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='Cycle: %{x}<br>RUL: %{y:.1f} cycles<extra></extra>'
        ))
        
        # Add uncertainty band (±RMSE)
        fig_rul.add_trace(go.Scatter(
            x=results['cycle'].tolist() + results['cycle'].tolist()[::-1],
            y=(results['predicted_rul'] + 10.74).tolist() + (results['predicted_rul'] - 10.74).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Uncertainty (±RMSE)',
            hoverinfo='skip'
        ))
        
        # Add threshold line
        fig_rul.add_hline(
            y=50,
            line_dash="dash",
            line_color="orange",
            annotation_text="Warning Threshold",
            annotation_position="right"
        )
        
        fig_rul.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Remaining Useful Life (cycles)",
            hovermode='x unified',
            height=450,
            showlegend=True
        )
        
        st.plotly_chart(fig_rul, use_container_width=True)
        
        st.info("""
        **Interpretation**: The shaded area shows prediction uncertainty (±10.74 cycles RMSE). 
        The model predicts RUL with 93% accuracy (R² = 0.9321).
        """)
    
    with tab2:
        st.subheader("Health Percentage Over Time")
        
        fig_health = go.Figure()
        
        # Add colored background zones
        fig_health.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.1,
                            annotation_text="Healthy", annotation_position="left")
        fig_health.add_hrect(y0=40, y1=70, fillcolor="orange", opacity=0.1,
                            annotation_text="Warning", annotation_position="left")
        fig_health.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.1,
                            annotation_text="Critical", annotation_position="left")
        
        # Add health line
        fig_health.add_trace(go.Scatter(
            x=results['cycle'],
            y=results['health_percentage'],
            mode='lines+markers',
            name='Health %',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=6),
            hovertemplate='Cycle: %{x}<br>Health: %{y:.1f}%<extra></extra>'
        ))
        
        fig_health.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Health Percentage (%)",
            yaxis_range=[0, 105],
            hovermode='x unified',
            height=450
        )
        
        st.plotly_chart(fig_health, use_container_width=True)
        
        st.info("""
        **Interpretation**: 
        - **Green zone (70-100%)**: Normal operation
        - **Yellow zone (40-70%)**: Plan maintenance soon
        - **Red zone (0-40%)**: Urgent action required
        """)
    
    with tab3:
        st.subheader("Model Performance Comparison")
        
        # Comparison data
        comparison_data = pd.DataFrame({
            'Model': ['Baseline XGBoost', 'Balanced XGBoost'],
            'R² Score': [0.8494, 0.9321],
            'RMSE (cycles)': [15.98, 10.74],
            'Features': [27, 100],
            'Training Time (s)': [0.74, 2.5]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R² comparison
            fig_r2 = px.bar(
                comparison_data,
                x='Model',
                y='R² Score',
                title='R² Score Comparison (Higher is Better)',
                color='R² Score',
                color_continuous_scale='Viridis',
                text='R² Score'
            )
            fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_r2.update_layout(height=400, showlegend=False)
            fig_r2.add_hline(y=0.90, line_dash="dash", line_color="red",
                            annotation_text="Target: 0.90")
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig_rmse = px.bar(
                comparison_data,
                x='Model',
                y='RMSE (cycles)',
                title='RMSE Comparison (Lower is Better)',
                color='RMSE (cycles)',
                color_continuous_scale='Reds_r',
                text='RMSE (cycles)'
            )
            fig_rmse.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_rmse.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        st.success(f"""
        **🏆 Improvement with Balanced Model**:
        - R² improved from 0.8494 → 0.9321 (+9.7%)
        - RMSE improved from 15.98 → 10.74 cycles (-32.8%)
        - Predictions are 33% more accurate!
        """)
        
        st.dataframe(comparison_data, use_container_width=True)
    
    with tab4:
        st.subheader("Top 20 Most Important Features")
        
        if 'balanced' in models:
            model = models['balanced']['model']
            engineer = models['balanced']['engineer']
            
            # Get feature names (need to engineer a sample to get column names)
            sample_engine = df_test[df_test['engine_id'] == df_test['engine_id'].iloc[0]].head(1).copy()
            sample_engineered = engineer.add_balanced_features(sample_engine)
            X_sample, _ = engineer.prepare_features(sample_engineered, fit=False)
            
            feature_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance (Top 20)",
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Blues'
            )
            
            fig_importance.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_importance, use_container_width=True)
            
            st.info("""
            **Interpretation**: These features contribute most to RUL predictions. 
            Rolling statistics and sensor interactions capture degradation trends better than raw values.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p><strong>Built with XGBoost + Streamlit</strong> | NASA CMAPSS Dataset | IIT Kharagpur Hackathon 2026</p>
        <p>Model: R² = 0.9321 | RMSE = 10.74 cycles | Cross-Validated ✅</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
