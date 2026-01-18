"""
COMPREHENSIVE MODEL COMPARISON FOR TIME-SERIES RUL PREDICTION
==============================================================
This script compares ALL viable approaches:
1. Traditional ML (XGBoost, Random Forest) - treating each cycle independently
2. Deep Learning Time Series (LSTM, GRU, 1D CNN) - leveraging temporal patterns
3. Transformer-based (Temporal Fusion Transformer) - state-of-the-art

WHY THIS MATTERS:
- Engine degradation is SEQUENTIAL - sensor values at time t depend on time t-1
- Traditional ML ignores this temporal dependency
- Deep learning can capture degradation trajectories
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Traditional ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except:
    HAS_TENSORFLOW = False


class DataPreprocessor:
    """Unified data preprocessing for all models"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.index_cols = ['engine_id', 'cycle']
        self.setting_cols = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        self.all_cols = self.index_cols + self.setting_cols + self.sensor_cols
        self.constant_features = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                                   'sensor_16', 'sensor_18', 'sensor_19', 'setting_3']
    
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
    
    def prepare_features(self, df, fit=True):
        df = df.copy()
        df = df.drop(columns=[c for c in self.constant_features if c in df.columns])
        feature_cols = [c for c in df.columns if c not in ['engine_id', 'RUL']]
        X = df[feature_cols]
        y = df['RUL'] if 'RUL' in df.columns else None
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        return X_scaled, y, df['engine_id']
    
    def create_sequences(self, df, sequence_length=50):
        """
        Create sequences for time-series models (LSTM, GRU, CNN)
        
        CRITICAL: This captures temporal dependencies
        Each sample is a window of 'sequence_length' consecutive cycles
        """
        df = df.copy()
        df = df.drop(columns=[c for c in self.constant_features if c in df.columns])
        feature_cols = [c for c in df.columns if c not in ['engine_id', 'cycle', 'RUL']]
        
        X_seq, y_seq = [], []
        
        for engine_id in df['engine_id'].unique():
            engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')
            features = self.scaler.transform(engine_data[feature_cols])
            rul = engine_data['RUL'].values
            
            # Create sliding windows
            for i in range(len(features) - sequence_length + 1):
                X_seq.append(features[i:i + sequence_length])
                y_seq.append(rul[i + sequence_length - 1])
        
        return np.array(X_seq), np.array(y_seq)


def evaluate_model(y_true, y_pred, model_name, training_time):
    """Unified evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time
    }


def build_lstm_model(sequence_length, n_features):
    """
    LSTM Model - Designed for Time Series
    
    WHY LSTM:
    - Has "memory" - remembers past sensor values
    - Captures degradation trends over time
    - Industry standard for predictive maintenance
    
    ARCHITECTURE:
    - 2 LSTM layers with dropout (prevent overfitting)
    - Dense layers for final prediction
    """
    model = keras.Sequential([
        layers.LSTM(100, return_sequences=True, input_shape=(sequence_length, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(50, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru_model(sequence_length, n_features):
    """
    GRU Model - Faster alternative to LSTM
    
    WHY GRU:
    - Similar to LSTM but simpler (fewer parameters)
    - Often performs as well as LSTM
    - Faster training
    """
    model = keras.Sequential([
        layers.GRU(100, return_sequences=True, input_shape=(sequence_length, n_features)),
        layers.Dropout(0.2),
        layers.GRU(50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(50, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_cnn1d_model(sequence_length, n_features):
    """
    1D CNN Model - Captures local temporal patterns
    
    WHY 1D CNN:
    - Detects patterns in sensor readings over time
    - Faster than LSTM/GRU
    - Good for detecting sudden changes
    """
    model = keras.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=(sequence_length, n_features)),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_hybrid_cnn_lstm(sequence_length, n_features):
    """
    Hybrid CNN-LSTM - Best of both worlds
    
    WHY HYBRID:
    - CNN extracts local features
    - LSTM captures long-term dependencies
    - Often achieves best performance
    """
    model = keras.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=(sequence_length, n_features)),
        layers.MaxPooling1D(2),
        layers.LSTM(50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(50, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    print("="*80)
    print("ULTIMATE MODEL COMPARISON - TIME SERIES vs TRADITIONAL ML")
    print("="*80)
    
    # Load data
    data_dir = "../data"
    train_path = os.path.join(data_dir, "train_FD001.txt")
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(train_path)
    df = preprocessor.compute_rul(df)
    
    print(f"\n📊 Dataset: {len(df)} samples, {df['engine_id'].nunique()} engines")
    
    results = []
    
    # ========================================================================
    # PART 1: TRADITIONAL ML (No Time Series)
    # ========================================================================
    print("\n" + "="*80)
    print("PART 1: TRADITIONAL ML (Treating each cycle independently)")
    print("="*80)
    
    X, y, engine_ids = preprocessor.prepare_features(df, fit=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost
    if HAS_XGBOOST:
        print("\n🔧 Training XGBoost...")
        start = time.time()
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        train_time = time.time() - start
        y_pred = xgb_model.predict(X_val)
        results.append(evaluate_model(y_val, y_pred, "XGBoost (No Time Series)", train_time))
        print(f"   RMSE: {results[-1]['rmse']:.2f}, Time: {train_time:.2f}s")
    
    # LightGBM
    if HAS_LIGHTGBM:
        print("\n🔧 Training LightGBM...")
        start = time.time()
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_train, y_train)
        train_time = time.time() - start
        y_pred = lgb_model.predict(X_val)
        results.append(evaluate_model(y_val, y_pred, "LightGBM (No Time Series)", train_time))
        print(f"   RMSE: {results[-1]['rmse']:.2f}, Time: {train_time:.2f}s")
    
    # Random Forest
    print("\n🔧 Training Random Forest...")
    start = time.time()
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = rf_model.predict(X_val)
    results.append(evaluate_model(y_val, y_pred, "Random Forest (No Time Series)", train_time))
    print(f"   RMSE: {results[-1]['rmse']:.2f}, Time: {train_time:.2f}s")
    
    # ========================================================================
    # PART 2: TIME SERIES MODELS (Leveraging Temporal Patterns)
    # ========================================================================
    if HAS_TENSORFLOW:
        print("\n" + "="*80)
        print("PART 2: TIME SERIES MODELS (Leveraging temporal dependencies)")
        print("="*80)
        
        sequence_length = 50
        print(f"\n📦 Creating sequences (window size: {sequence_length} cycles)...")
        
        # Fit scaler on full training data first
        preprocessor.prepare_features(df, fit=True)
        X_seq, y_seq = preprocessor.create_sequences(df, sequence_length)
        X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        print(f"   Sequence shape: {X_train_seq.shape}")
        n_features = X_train_seq.shape[2]
        
        # LSTM
        print("\n🔧 Training LSTM...")
        start = time.time()
        lstm_model = build_lstm_model(sequence_length, n_features)
        lstm_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=256, 
                      validation_data=(X_val_seq, y_val_seq), verbose=0)
        train_time = time.time() - start
        y_pred = lstm_model.predict(X_val_seq, verbose=0).flatten()
        results.append(evaluate_model(y_val_seq, y_pred, "LSTM (Time Series)", train_time))
        print(f"   RMSE: {results[-1]['rmse']:.2f}, Time: {train_time:.2f}s")
        
        # GRU
        print("\n🔧 Training GRU...")
        start = time.time()
        gru_model = build_gru_model(sequence_length, n_features)
        gru_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=256,
                     validation_data=(X_val_seq, y_val_seq), verbose=0)
        train_time = time.time() - start
        y_pred = gru_model.predict(X_val_seq, verbose=0).flatten()
        results.append(evaluate_model(y_val_seq, y_pred, "GRU (Time Series)", train_time))
        print(f"   RMSE: {results[-1]['rmse']:.2f}, Time: {train_time:.2f}s")
        
        # 1D CNN
        print("\n🔧 Training 1D CNN...")
        start = time.time()
        cnn_model = build_cnn1d_model(sequence_length, n_features)
        cnn_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=256,
                     validation_data=(X_val_seq, y_val_seq), verbose=0)
        train_time = time.time() - start
        y_pred = cnn_model.predict(X_val_seq, verbose=0).flatten()
        results.append(evaluate_model(y_val_seq, y_pred, "1D CNN (Time Series)", train_time))
        print(f"   RMSE: {results[-1]['rmse']:.2f}, Time: {train_time:.2f}s")
        
        # Hybrid CNN-LSTM
        print("\n🔧 Training Hybrid CNN-LSTM...")
        start = time.time()
        hybrid_model = build_hybrid_cnn_lstm(sequence_length, n_features)
        hybrid_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=256,
                        validation_data=(X_val_seq, y_val_seq), verbose=0)
        train_time = time.time() - start
        y_pred = hybrid_model.predict(X_val_seq, verbose=0).flatten()
        results.append(evaluate_model(y_val_seq, y_pred, "CNN-LSTM Hybrid (Time Series)", train_time))
        print(f"   RMSE: {results[-1]['rmse']:.2f}, Time: {train_time:.2f}s")
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS - SORTED BY RMSE (Lower is Better)")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    
    print("\n" + results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]
    print(f"\n🥇 WINNER: {best_model['model_name']}")
    print(f"   RMSE: {best_model['rmse']:.2f} cycles")
    print(f"   R²: {best_model['r2']:.4f}")
    print(f"   Training Time: {best_model['training_time']:.2f}s")
    
    # Save results
    results_df.to_csv("ultimate_model_comparison.csv", index=False)
    print(f"\n💾 Results saved to: ultimate_model_comparison.csv")
    
    # Recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATION FOR HACKATHON")
    print("="*80)
    
    if "Time Series" in best_model['model_name']:
        print(f"""
✅ USE TIME SERIES MODEL: {best_model['model_name']}

WHY:
- Achieves best RMSE ({best_model['rmse']:.2f} cycles)
- Leverages temporal dependencies in sensor data
- More sophisticated approach (impresses judges)
- Captures degradation trajectories over time

TRADEOFF:
- Longer training time ({best_model['training_time']:.1f}s vs <1s for XGBoost)
- More complex to explain
- Requires sequence preprocessing

JUDGE EXPLANATION:
"Engine degradation is a TIME SERIES problem. Sensor values at time t depend on 
previous values. Our {best_model['model_name']} model captures these temporal patterns,
achieving {best_model['rmse']:.2f} RMSE compared to {results_df[results_df['model_name'].str.contains('XGBoost')]['rmse'].values[0]:.2f} for XGBoost."
        """)
    else:
        print(f"""
✅ USE TRADITIONAL ML: {best_model['model_name']}

WHY:
- Achieves best RMSE ({best_model['rmse']:.2f} cycles)
- Extremely fast training ({best_model['training_time']:.2f}s)
- Simple and explainable
- Easier to debug during demo

TRADEOFF:
- Doesn't explicitly model time dependencies
- May miss subtle degradation patterns

JUDGE EXPLANATION:
"While this is time-series data, {best_model['model_name']} achieves excellent 
performance ({best_model['rmse']:.2f} RMSE) by treating each cycle independently.
The speed and simplicity make it ideal for production deployment."
        """)


if __name__ == "__main__":
    main()
