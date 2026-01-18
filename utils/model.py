"""
XGBoost Model Training Module
==============================
This module implements the XGBoost Regressor for RUL prediction.

WHY XGBOOST?
1. Handles non-linear sensor degradation patterns
2. Robust to outliers (common in industrial sensors)
3. Fast training and inference (critical for live demos)
4. Built-in regularization prevents overfitting
5. Provides feature importance (explainability)

WHAT IS BOOSTING?
- Builds trees sequentially
- Each tree corrects errors from previous trees
- Final prediction = weighted sum of all trees
- Think: "Learning from mistakes"

WHY IT WORKS FOR ENGINES:
- Sensor degradation is complex and non-linear
- Different sensors degrade at different rates
- XGBoost learns these patterns through iterative refinement
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib


class RULPredictor:
    """
    XGBoost-based Remaining Useful Life Predictor
    
    DESIGN: Wraps XGBoost in a class for clean interface
    """
    
    def __init__(self, params=None):
        """
        Initialize XGBoost model with sensible defaults
        
        HYPERPARAMETER CHOICES (Explainable to Judges):
        
        n_estimators=100:
            - Number of boosting rounds (trees)
            - More trees = better fit, but slower
            - 100 is a good balance for this dataset size
        
        max_depth=6:
            - Maximum depth of each tree
            - Deeper = more complex patterns, but risk overfitting
            - 6 is standard for tabular data
        
        learning_rate=0.1:
            - How much each tree contributes
            - Lower = more conservative, needs more trees
            - 0.1 is standard starting point
        
        subsample=0.8:
            - Fraction of samples used per tree
            - Prevents overfitting by introducing randomness
            - 0.8 means each tree sees 80% of data
        
        colsample_bytree=0.8:
            - Fraction of features used per tree
            - Similar to Random Forest's feature bagging
            - Improves generalization
        
        random_state=42:
            - Ensures reproducibility (same results every run)
            - Critical for debugging and comparison
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1  # Use all CPU cores
            }
        
        self.model = xgb.XGBRegressor(**params)
        self.params = params
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the XGBoost model
        
        HOW BOOSTING WORKS (Explain to Judges):
        1. Start with a simple prediction (mean RUL)
        2. Build a tree to predict the errors
        3. Add this tree's predictions to improve overall prediction
        4. Repeat 100 times (n_estimators)
        5. Each iteration focuses on samples with high error
        
        WHY THIS HELPS:
        - Early trees learn general patterns
        - Later trees learn subtle, complex patterns
        - Combination is more accurate than any single tree
        
        Args:
            X_train: Training features
            y_train: Training RUL values
            X_val: Validation features (optional, for monitoring)
            y_val: Validation RUL values (optional)
        """
        print("\n" + "="*60)
        print("STEP 2: MODEL TRAINING (XGBoost)")
        print("="*60)
        
        print("\n🏗️  Building XGBoost Regressor...")
        print(f"   Hyperparameters: {self.params}")
        
        # Train the model
        # WHY eval_set: Monitors validation performance during training
        # WHY verbose=False: Cleaner output for demo
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print("✅ Training complete!")
        
        # Evaluate on training and validation sets
        self._evaluate(X_train, y_train, X_val, y_val)
    
    def _evaluate(self, X_train, y_train, X_val, y_val):
        """
        Evaluate model performance
        
        METRICS EXPLAINED:
        
        RMSE (Root Mean Squared Error):
            - Average prediction error in cycles
            - Penalizes large errors more than small ones
            - Lower is better
            - EXAMPLE: RMSE=15 means predictions are off by ~15 cycles on average
        
        MAE (Mean Absolute Error):
            - Average absolute prediction error
            - More intuitive than RMSE
            - Less sensitive to outliers
        
        R² Score:
            - Proportion of variance explained (0 to 1)
            - 1.0 = perfect predictions
            - 0.0 = model is no better than predicting the mean
            - EXAMPLE: R²=0.85 means model explains 85% of RUL variation
        """
        print("\n📊 MODEL EVALUATION")
        print("-" * 60)
        
        # Training metrics
        y_train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        print(f"Training Set:")
        print(f"  RMSE: {train_rmse:.2f} cycles")
        print(f"  R² Score: {train_r2:.4f}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            
            print(f"\nValidation Set:")
            print(f"  RMSE: {val_rmse:.2f} cycles")
            print(f"  MAE: {val_mae:.2f} cycles")
            print(f"  R² Score: {val_r2:.4f}")
            
            # Interpretation for judges
            print(f"\n💡 INTERPRETATION:")
            print(f"  - Model predicts RUL with ~{val_rmse:.0f} cycle accuracy")
            print(f"  - Explains {val_r2*100:.1f}% of RUL variation")
            
            # Check for overfitting
            if train_rmse < val_rmse * 0.7:
                print(f"  ⚠️  Warning: Possible overfitting (train RMSE much lower than val)")
            else:
                print(f"  ✅ Good generalization (train/val RMSE are close)")
    
    def predict(self, X):
        """
        Predict RUL for new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted RUL values (numpy array)
        """
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores
        
        WHY THIS MATTERS:
        - Shows which sensors are most predictive
        - Helps engineers understand degradation patterns
        - Validates that model is learning sensible patterns
        
        Returns:
            DataFrame with features and importance scores
        """
        import pandas as pd
        
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        print(f"📂 Model loaded from {filepath}")


def explain_xgboost():
    """
    One-paragraph explanation for judges
    """
    explanation = """
    XGBoost (Extreme Gradient Boosting) is an ensemble learning method that builds 
    multiple decision trees sequentially. Each tree learns to correct the errors made 
    by previous trees, gradually improving predictions. For jet engine maintenance, 
    this is ideal because sensor degradation patterns are complex and non-linear - 
    different sensors degrade at different rates and interact in subtle ways. XGBoost 
    automatically learns these patterns without manual feature engineering. The model 
    helps operators by converting raw sensor readings into actionable predictions: 
    "This engine has 50 cycles remaining" or "Health is at 40% - schedule maintenance soon."
    This prevents unexpected failures and optimizes maintenance schedules.
    """
    return explanation


if __name__ == "__main__":
    print(explain_xgboost())
