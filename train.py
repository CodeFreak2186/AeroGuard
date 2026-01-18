"""
Training Script - Main Entry Point
===================================
This script orchestrates the complete training pipeline.

RUN THIS TO TRAIN THE MODEL:
    python train.py

OUTPUT:
    - models/xgboost_model.pkl
    - models/scaler.pkl
    - Console output with RMSE and R² scores
"""

import os
import sys

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from data_loader import preprocess_training_data
from model import RULPredictor, explain_xgboost


def main():
    """
    Main training pipeline
    
    WORKFLOW:
    1. Load and preprocess data
    2. Train XGBoost model
    3. Evaluate performance
    4. Save model and scaler
    """
    print("\n" + "="*60)
    print("🚀 PREDICTIVE MAINTENANCE SYSTEM - TRAINING")
    print("   IIT Kharagpur Hackathon (Kshitij)")
    print("="*60)
    
    # Paths
    data_dir = "data"
    models_dir = "models"
    train_path = os.path.join(data_dir, "train_FD001.txt")
    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(train_path):
        print(f"\n❌ ERROR: Training data not found at {train_path}")
        print("\nPlease ensure train_FD001.txt is in the 'data' folder.")
        print("Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        return
    
    # STEP 1: Data Preprocessing
    X_train, X_val, y_train, y_val, data_loader = preprocess_training_data(train_path)
    
    # STEP 2: Model Training
    predictor = RULPredictor()
    predictor.train(X_train, y_train, X_val, y_val)
    
    # STEP 3: Feature Importance (Optional - for judges)
    print("\n📊 TOP 5 MOST IMPORTANT FEATURES:")
    print("-" * 60)
    feature_importance = predictor.get_feature_importance(X_train.columns)
    print(feature_importance.head(10).to_string(index=False))
    
    print("\n💡 INTERPRETATION:")
    print("   These sensors show the strongest correlation with engine degradation.")
    print("   Operators should monitor these closely for early failure detection.")
    
    # STEP 4: Save Model and Scaler
    print("\n" + "="*60)
    print("STEP 3: SAVING MODEL ARTIFACTS")
    print("="*60)
    predictor.save(model_path)
    data_loader.save_scaler(scaler_path)
    
    # Final Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print("\nNext steps:")
    print("  1. Run the dashboard: streamlit run app.py")
    print("  2. Select any engine ID to see predictions")
    print("  3. Explain to judges how XGBoost learns degradation patterns")
    
    # Print explanation for judges
    print("\n" + "="*60)
    print("MODEL EXPLANATION (For Judges)")
    print("="*60)
    print(explain_xgboost())


if __name__ == "__main__":
    main()
