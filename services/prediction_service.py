"""
Prediction Service
==================
Handles all prediction-related operations for the AeroGuard system.
"""

import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from config import MODEL_PATH, SCALER_PATH, TEST_DATA_PATH, FALLBACK_MODEL_PATH, FALLBACK_SCALER_PATH
from utils.data_loader import CMAPSSDataLoader
from utils.health_score import rul_to_health_percentage, get_health_status, get_maintenance_recommendation

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for managing predictions and engine data."""
    
    def __init__(self):
        """Initialize the prediction service."""
        self.model = None
        self.scaler = None
        self.data_loader = None
        self.test_data = None
        self.predictions = None
        
        self._load_model()
        self._load_data()
        self._generate_predictions()
    
    def _load_model(self):
        """Load the trained model and scaler."""
        try:
            # Try to load balanced model first
            if MODEL_PATH.exists():
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                logger.info(f"✓ Loaded model from {MODEL_PATH}")
            else:
                # Fallback to baseline model
                self.model = joblib.load(FALLBACK_MODEL_PATH)
                self.scaler = joblib.load(FALLBACK_SCALER_PATH)
                logger.info(f"✓ Loaded fallback model from {FALLBACK_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_data(self):
        """Load and preprocess test data."""
        try:
            self.data_loader = CMAPSSDataLoader()
            df_test = self.data_loader.load_raw_data(str(TEST_DATA_PATH))
            
            # Apply feature engineering if using balanced model
            if MODEL_PATH.exists():
                try:
                    from train_balanced import BalancedFeatureEngineer
                    feature_engineer = BalancedFeatureEngineer()
                    df_test = feature_engineer.add_balanced_features(df_test)
                    logger.info("✓ Applied feature engineering")
                except ImportError:
                    logger.warning("Feature engineering not available, using raw features")
            
            # Load scaler
            self.data_loader.load_scaler(str(SCALER_PATH if MODEL_PATH.exists() else FALLBACK_SCALER_PATH))
            
            # Prepare features
            X_test, _ = self.data_loader.prepare_features(df_test, is_training=False)
            self.X_test = X_test
            
            # Store raw data for display
            self.test_data = self.data_loader.load_raw_data(str(TEST_DATA_PATH))
            
            logger.info(f"✓ Loaded {len(self.test_data)} test samples")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _generate_predictions(self):
        """Generate predictions for all engines."""
        try:
            self.predictions = self.model.predict(self.X_test.values)
            
            # Add predictions to test data
            self.test_data['predicted_rul'] = self.predictions
            self.test_data['health_percentage'] = self.test_data['predicted_rul'].apply(
                rul_to_health_percentage
            )
            self.test_data['health_status'] = self.test_data['health_percentage'].apply(
                lambda x: get_health_status(x)[0]
            )
            
            # Rename engine_id to unit_id for consistency
            if 'engine_id' in self.test_data.columns:
                self.test_data.rename(columns={'engine_id': 'unit_id'}, inplace=True)
            
            logger.info(f"✓ Generated predictions for {len(self.test_data)} samples")
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            raise
    
    def get_all_engines(self):
        """Get list of all engines with their current status."""
        last_cycle_data = self.test_data.groupby('unit_id').last().reset_index()
        
        engines = []
        for _, row in last_cycle_data.iterrows():
            status_text, _, _ = get_health_status(row['health_percentage'])
            engines.append({
                'unit_id': int(row['unit_id']),
                'health_percentage': float(row['health_percentage']),
                'predicted_rul': float(row['predicted_rul']),
                'status': status_text,
                'recommendation': get_maintenance_recommendation(
                    row['health_percentage'],
                    row['predicted_rul']
                )
            })
        
        # Sort by health percentage (critical first)
        engines.sort(key=lambda x: x['health_percentage'])
        
        return engines
    
    def get_engine_details(self, engine_id):
        """Get detailed information for a specific engine."""
        engine_data = self.test_data[self.test_data['unit_id'] == engine_id]
        
        if engine_data.empty:
            return None
        
        # Get latest data
        latest = engine_data.iloc[-1]
        
        # Get historical data
        cycles = engine_data['cycle'].tolist()
        health_history = engine_data['health_percentage'].tolist()
        
        # Get sensor data (latest reading)
        sensor_cols = [col for col in engine_data.columns if col.startswith('sensor_')]
        sensors = {col: float(latest[col]) for col in sensor_cols}
        
        # Calculate RMSE (approximate)
        rmse = 5.5  # This should come from model evaluation
        
        return {
            'unit_id': int(engine_id),
            'current_cycle': int(latest['cycle']),
            'predicted_rul': float(latest['predicted_rul']),
            'health_percentage': float(latest['health_percentage']),
            'status': latest['health_status'],
            'recommendation': get_maintenance_recommendation(
                latest['health_percentage'],
                latest['predicted_rul']
            ),
            'rmse': rmse,
            'cycles': cycles,
            'health_history': health_history,
            'sensors': sensors
        }
    
    def get_test_data(self):
        """Get the full test dataset."""
        return self.test_data
