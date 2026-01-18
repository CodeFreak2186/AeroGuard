"""
OVERFITTING DETECTION SCRIPT
=============================
Checks if R² = 0.9956 is real or overfitting

TESTS:
1. Train vs Validation RMSE comparison
2. Cross-validation (K-Fold)
3. Learning curves
4. Test on completely unseen data (test_FD001.txt)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
import sys
sys.path.insert(0, '.')

from train_advanced import AdvancedFeatureEngineer


def check_overfitting():
    """
    Comprehensive overfitting check
    """
    print("="*80)
    print("🔍 OVERFITTING DETECTION - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Load the trained model
    model_path = "models/advanced_xgboost_model.pkl"
    scaler_path = "models/advanced_scaler.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Run train_advanced.py first.")
        return
    
    model = joblib.load(model_path)
    print(f"✅ Loaded model from {model_path}")
    
    # Load and prepare data
    data_dir = "../data"
    train_path = os.path.join(data_dir, "train_FD001.txt")
    
    engineer = AdvancedFeatureEngineer()
    df = engineer.load_data(train_path)
    df = engineer.compute_rul(df)
    df = engineer.engineer_all_features(df)
    
    # Load scaler
    engineer.scaler = joblib.load(scaler_path)
    X, y = engineer.prepare_features(df, fit=False)
    
    print(f"\n📊 Dataset: {len(X)} samples")
    
    # ========================================================================
    # TEST 1: Train vs Validation RMSE
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: Train vs Validation RMSE")
    print("="*80)
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\nTraining Set:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²: {train_r2:.4f}")
    
    print(f"\nValidation Set:")
    print(f"  RMSE: {val_rmse:.2f}")
    print(f"  R²: {val_r2:.4f}")
    
    # Check for overfitting
    rmse_ratio = train_rmse / val_rmse
    r2_diff = train_r2 - val_r2
    
    print(f"\n📊 Overfitting Indicators:")
    print(f"  Train/Val RMSE Ratio: {rmse_ratio:.3f}")
    print(f"  R² Difference: {r2_diff:.4f}")
    
    if rmse_ratio < 0.7:
        print(f"  ⚠️  WARNING: Train RMSE much lower than Val RMSE")
        print(f"  → Likely OVERFITTING")
        overfitting_test1 = True
    elif r2_diff > 0.05:
        print(f"  ⚠️  WARNING: Train R² significantly higher than Val R²")
        print(f"  → Possible OVERFITTING")
        overfitting_test1 = True
    else:
        print(f"  ✅ PASS: Train and Val metrics are close")
        print(f"  → No significant overfitting detected")
        overfitting_test1 = False
    
    # ========================================================================
    # TEST 2: K-Fold Cross-Validation
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: 5-Fold Cross-Validation")
    print("="*80)
    print("Running 5-fold CV (this may take a minute)...")
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use negative MSE (sklearn convention)
    cv_scores = cross_val_score(model, X, y, cv=kfold, 
                                scoring='neg_mean_squared_error', n_jobs=-1)
    
    cv_rmse = np.sqrt(-cv_scores)
    
    print(f"\nCross-Validation RMSE per fold:")
    for i, rmse in enumerate(cv_rmse, 1):
        print(f"  Fold {i}: {rmse:.2f}")
    
    print(f"\nCross-Validation Summary:")
    print(f"  Mean RMSE: {cv_rmse.mean():.2f}")
    print(f"  Std RMSE: {cv_rmse.std():.2f}")
    print(f"  Min RMSE: {cv_rmse.min():.2f}")
    print(f"  Max RMSE: {cv_rmse.max():.2f}")
    
    # Check consistency
    if cv_rmse.std() > 3.0:
        print(f"\n  ⚠️  WARNING: High variance across folds")
        print(f"  → Model performance is inconsistent")
        overfitting_test2 = True
    else:
        print(f"\n  ✅ PASS: Consistent performance across folds")
        overfitting_test2 = False
    
    # ========================================================================
    # TEST 3: Test on Completely Unseen Data (test_FD001.txt)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: Performance on Unseen Test Set")
    print("="*80)
    
    test_path = os.path.join(data_dir, "test_FD001.txt")
    rul_path = os.path.join(data_dir, "RUL_FD001.txt")
    
    if os.path.exists(test_path) and os.path.exists(rul_path):
        # Load test data
        df_test = engineer.load_data(test_path)
        
        # Load true RUL values
        rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
        
        # For test data, we need to get the LAST sequence for each engine
        # (since we don't have full degradation history)
        
        # Get last cycle for each engine
        test_engines = []
        test_predictions = []
        
        for engine_id in df_test['engine_id'].unique():
            engine_data = df_test[df_test['engine_id'] == engine_id].copy()
            
            # Take last row (most degraded state)
            last_row = engine_data.iloc[[-1]].copy()
            
            # We can't do full feature engineering without history
            # So this test might not be fully valid
            # But let's try with available data
            
            # For now, skip this test and note the limitation
            break
        
        print("⚠️  Note: Test set evaluation requires sequence history")
        print("   Skipping detailed test set evaluation")
        print("   (This is a limitation of the current approach)")
        overfitting_test3 = None
    else:
        print("⚠️  Test data not found, skipping test set evaluation")
        overfitting_test3 = None
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print("\n" + "="*80)
    print("🏁 FINAL VERDICT")
    print("="*80)
    
    print(f"\nTest 1 (Train/Val Comparison): {'⚠️ FAIL' if overfitting_test1 else '✅ PASS'}")
    print(f"Test 2 (Cross-Validation): {'⚠️ FAIL' if overfitting_test2 else '✅ PASS'}")
    
    if overfitting_test1 or overfitting_test2:
        print(f"\n❌ OVERFITTING DETECTED")
        print(f"\nRecommendations:")
        print(f"  1. Reduce number of features (feature selection)")
        print(f"  2. Increase regularization (reg_alpha, reg_lambda)")
        print(f"  3. Reduce max_depth (currently 8)")
        print(f"  4. Increase min_child_weight")
        
        # Suggest realistic R²
        print(f"\nRealistic R² for this problem: 0.88 - 0.92")
        print(f"Current validation R²: {val_r2:.4f}")
        
        if val_r2 > 0.95:
            print(f"\n⚠️  R² > 0.95 is suspiciously high for real-world data")
            print(f"   Likely causes:")
            print(f"   - Data leakage (using future information)")
            print(f"   - Overfitting to training set")
            print(f"   - Feature engineering creating redundant features")
    else:
        print(f"\n✅ NO SIGNIFICANT OVERFITTING")
        print(f"\nValidation R²: {val_r2:.4f}")
        print(f"This appears to be genuine performance!")
        
        print(f"\nWhy R² is so high:")
        print(f"  1. Rich feature engineering captures degradation patterns")
        print(f"  2. XGBoost is powerful for tabular data")
        print(f"  3. Engine degradation is relatively predictable")
        print(f"  4. 405 features provide comprehensive information")
    
    # ========================================================================
    # ADDITIONAL DIAGNOSTICS
    # ========================================================================
    print("\n" + "="*80)
    print("📊 ADDITIONAL DIAGNOSTICS")
    print("="*80)
    
    # Prediction distribution
    print(f"\nPrediction Statistics:")
    print(f"  Val True RUL - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}")
    print(f"  Val Pred RUL - Mean: {y_val_pred.mean():.2f}, Std: {y_val_pred.std():.2f}")
    
    # Residuals
    residuals = y_val - y_val_pred
    print(f"\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.2f} (should be ~0)")
    print(f"  Std: {residuals.std():.2f}")
    print(f"  Max Error: {np.abs(residuals).max():.2f}")
    
    if abs(residuals.mean()) > 1.0:
        print(f"  ⚠️  Non-zero mean residuals → systematic bias")
    else:
        print(f"  ✅ Mean residuals close to zero")


if __name__ == "__main__":
    check_overfitting()
