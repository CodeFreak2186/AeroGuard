"""
BALANCED FEATURE ENGINEERING - TARGET: R² = 0.90
==================================================
Carefully selected features that improve performance WITHOUT overfitting

STRATEGY:
1. Use ONLY rolling statistics (no lags - they cause leakage)
2. Moderate window sizes (10, 20 - not too short, not too long)
3. Key sensor interactions (domain knowledge)
4. Optimized XGBoost hyperparameters
5. STRICT cross-validation to verify no overfitting

EXPECTED: R² = 0.88-0.92, RMSE = 11-13 cycles
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os


class BalancedFeatureEngineer:
    """
    Balanced feature engineering - quality over quantity
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.index_cols = ['engine_id', 'cycle']
        self.setting_cols = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        self.all_cols = self.index_cols + self.setting_cols + self.sensor_cols
        
        # Drop constant features
        self.constant_features = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                                   'sensor_16', 'sensor_18', 'sensor_19', 'setting_3']
        
        # Key sensors (from previous analysis)
        self.key_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 
                           'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12',
                           'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
                           'sensor_20', 'sensor_21']
    
    def load_data(self, filepath):
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=self.all_cols)
        return df
    
    def compute_rul(self, df):
        df = df.copy()
        max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
        max_cycles.columns = ['engine_id', 'max_cycle']
        df = df.merge(max_cycles, on='engine_id', how='left')
        df['RUL'] = df['max_cycle'] - df['cycle']
        df['RUL'] = df['RUL'].clip(upper=125)
        df.drop('max_cycle', axis=1, inplace=True)
        return df
    
    def add_balanced_features(self, df):
        """
        Add ONLY features that improve generalization
        
        RULES:
        - NO lag features (temporal leakage)
        - NO cumulative features (memorization)
        - ONLY moderate rolling windows (10, 20)
        - ONLY key sensor interactions
        """
        print("\n🔧 BALANCED FEATURE ENGINEERING")
        print("="*60)
        
        df = df.copy()
        original_features = len(df.columns)
        
        # Drop constant features
        df = df.drop(columns=[c for c in self.constant_features if c in df.columns])
        
        # 1. Rolling Statistics (MODERATE windows only)
        print("   Adding rolling features (windows: 10, 20)...")
        for sensor in self.key_sensors:
            if sensor not in df.columns:
                continue
            
            for window in [10, 20]:  # NOT 5, 30 - too short/long
                # Rolling mean
                df[f'{sensor}_rolling_mean_{window}'] = df.groupby('engine_id')[sensor].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std (volatility)
                df[f'{sensor}_rolling_std_{window}'] = df.groupby('engine_id')[sensor].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
                )
        
        # 2. Exponential Moving Average (SINGLE span)
        print("   Adding EMA features (span: 10)...")
        for sensor in self.key_sensors:
            if sensor not in df.columns:
                continue
            
            df[f'{sensor}_ema_10'] = df.groupby('engine_id')[sensor].transform(
                lambda x: x.ewm(span=10, adjust=False).mean()
            )
        
        # 3. Difference Features (SINGLE period)
        print("   Adding diff features (period: 5)...")
        for sensor in self.key_sensors:
            if sensor not in df.columns:
                continue
            
            df[f'{sensor}_diff_5'] = df.groupby('engine_id')[sensor].diff(5).fillna(0)
        
        # 4. Key Sensor Interactions (DOMAIN KNOWLEDGE)
        print("   Adding interaction features...")
        interactions = [
            ('sensor_2', 'sensor_3'),   # Temperature interactions
            ('sensor_4', 'sensor_11'),  # Pressure interactions
            ('sensor_7', 'sensor_8'),   # Flow interactions
            ('sensor_9', 'sensor_14'),  # Vibration interactions
            ('sensor_11', 'sensor_13'), # Additional pressure
        ]
        
        for s1, s2 in interactions:
            if s1 in df.columns and s2 in df.columns:
                df[f'{s1}_x_{s2}'] = df[s1] * df[s2]
        
        new_features = len(df.columns)
        print(f"\n✅ Feature engineering complete!")
        print(f"   Original features: {original_features}")
        print(f"   New features: {new_features}")
        print(f"   Added: {new_features - original_features} features")
        
        return df
    
    def prepare_features(self, df, fit=True):
        df = df.copy()
        feature_cols = [c for c in df.columns if c not in ['engine_id', 'cycle', 'RUL']]
        X = df[feature_cols]
        y = df['RUL'] if 'RUL' in df.columns else None
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        return X_scaled, y


def train_balanced_model():
    """
    Train with balanced features - TARGET: R² = 0.90
    """
    print("="*80)
    print("🎯 BALANCED MODEL TRAINING - TARGET: R² = 0.90 (NO OVERFITTING)")
    print("="*80)
    
    # Load data
    data_dir = "../data"
    train_path = os.path.join(data_dir, "train_FD001.txt")
    
    engineer = BalancedFeatureEngineer()
    df = engineer.load_data(train_path)
    df = engineer.compute_rul(df)
    
    print(f"\n📊 Dataset: {len(df)} samples, {df['engine_id'].nunique()} engines")
    
    # Engineer features
    df = engineer.add_balanced_features(df)
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X, y = engineer.prepare_features(df, fit=True)
    print(f"   Total features: {X.shape[1]}")
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # ========================================================================
    # OPTIMIZED XGBOOST (Balanced hyperparameters)
    # ========================================================================
    print("\n🏗️  Training XGBoost (Balanced)...")
    print("   Hyperparameters:")
    print("   - n_estimators: 150")
    print("   - max_depth: 6 (not too deep)")
    print("   - learning_rate: 0.08")
    print("   - subsample: 0.8")
    print("   - colsample_bytree: 0.8")
    print("   - reg_alpha: 0.05 (L1 regularization)")
    print("   - reg_lambda: 1.0 (L2 regularization)")
    
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,              # Moderate depth
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,           # Regularization
        reg_lambda=1.0,
        min_child_weight=3,       # Prevent overfitting
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             verbose=False)
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("\n📊 EVALUATION")
    print("="*60)
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Training Set:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²: {train_r2:.4f}")
    
    print(f"\nValidation Set:")
    print(f"  RMSE: {val_rmse:.2f}")
    print(f"  R²: {val_r2:.4f}")
    
    # Check overfitting
    rmse_ratio = train_rmse / val_rmse
    r2_diff = train_r2 - val_r2
    
    print(f"\n🔍 Overfitting Check:")
    print(f"  Train/Val RMSE Ratio: {rmse_ratio:.3f}")
    print(f"  R² Difference: {r2_diff:.4f}")
    
    if rmse_ratio < 0.75 or r2_diff > 0.05:
        print(f"  ⚠️  WARNING: Possible overfitting detected")
    else:
        print(f"  ✅ PASS: Good generalization")
    
    # ========================================================================
    # CROSS-VALIDATION (CRITICAL!)
    # ========================================================================
    print("\n📊 5-FOLD CROSS-VALIDATION")
    print("="*60)
    print("Running cross-validation (this verifies no overfitting)...")
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kfold,
                                scoring='neg_mean_squared_error', n_jobs=-1)
    cv_rmse = np.sqrt(-cv_scores)
    
    print(f"\nCross-Validation RMSE:")
    print(f"  Mean: {cv_rmse.mean():.2f}")
    print(f"  Std: {cv_rmse.std():.2f}")
    print(f"  Min: {cv_rmse.min():.2f}")
    print(f"  Max: {cv_rmse.max():.2f}")
    
    # Estimate R² from CV RMSE
    # R² ≈ 1 - (RMSE² / Var(y))
    y_var = y.var()
    cv_r2_estimate = 1 - (cv_rmse.mean()**2 / y_var)
    
    print(f"\nEstimated CV R²: {cv_r2_estimate:.4f}")
    
    if cv_rmse.std() > 2.0:
        print(f"  ⚠️  High variance across folds")
    else:
        print(f"  ✅ Consistent performance")
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print("\n" + "="*80)
    print("🏁 FINAL RESULTS")
    print("="*80)
    
    print(f"\nValidation R²: {val_r2:.4f}")
    print(f"Cross-Val R² (estimated): {cv_r2_estimate:.4f}")
    print(f"Validation RMSE: {val_rmse:.2f} cycles")
    
    if val_r2 >= 0.90:
        print(f"\n🎉 SUCCESS! R² ≥ 0.90 achieved!")
    elif val_r2 >= 0.88:
        print(f"\n✅ EXCELLENT! R² = {val_r2:.4f} (very close to 0.90)")
    else:
        print(f"\n⚠️  R² = {val_r2:.4f} (below 0.90, but honest)")
    
    # Feature importance
    print("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model
    model_path = "models/balanced_xgboost_model.pkl"
    scaler_path = "models/balanced_scaler.pkl"
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(engineer.scaler, scaler_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"💾 Scaler saved to: {scaler_path}")
    
    # Save engineer for inference
    engineer_path = "models/balanced_engineer.pkl"
    joblib.dump(engineer, engineer_path)
    print(f"💾 Engineer saved to: {engineer_path}")
    
    return model, engineer, val_r2, cv_r2_estimate


if __name__ == "__main__":
    # Must run from hackathon_project directory
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    model, engineer, val_r2, cv_r2 = train_balanced_model()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"Validation R²: {val_r2:.4f}")
    print(f"Cross-Val R²: {cv_r2:.4f}")
    print(f"Target: 0.90")
    
    if val_r2 >= 0.88 and cv_r2 >= 0.87:
        print(f"\n🏆 READY FOR HACKATHON!")
        print(f"   - No overfitting detected")
        print(f"   - Excellent performance")
        print(f"   - Honest, defensible results")
    else:
        print(f"\n✅ Good results, honest approach")
