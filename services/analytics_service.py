"""
Analytics Service
=================
Handles analytics and statistical computations for the dashboard.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime

from utils.health_score import get_health_status

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for generating analytics and statistics."""
    
    def __init__(self, prediction_service):
        """
        Initialize analytics service.
        
        Args:
            prediction_service: Instance of PredictionService
        """
        self.prediction_service = prediction_service
    
    def get_overview(self):
        """Get overview statistics for the fleet."""
        test_data = self.prediction_service.get_test_data()
        last_cycle_data = test_data.groupby('unit_id').last()
        
        # Get status counts
        statuses = last_cycle_data['health_percentage'].apply(
            lambda x: get_health_status(x)[0]
        )
        critical_count = sum(statuses == 'Critical')
        warning_count = sum(statuses == 'Warning')
        healthy_count = sum(statuses == 'Healthy')
        
        # Calculate averages
        avg_health = last_cycle_data['health_percentage'].mean()
        avg_rul = last_cycle_data['predicted_rul'].mean()
        
        return {
            'total_engines': len(last_cycle_data),
            'critical': int(critical_count),
            'warning': int(warning_count),
            'healthy': int(healthy_count),
            'avg_health': float(avg_health),
            'avg_rul': float(avg_rul),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_analytics(self):
        """Get detailed analytics for charts and visualizations."""
        test_data = self.prediction_service.get_test_data()
        
        # RUL distribution
        last_cycle_data = test_data.groupby('unit_id').last()
        rul_bins = [0, 25, 50, 75, 100, 125]
        rul_labels = ['0-25', '25-50', '50-75', '75-100', '100-125']
        rul_dist = pd.cut(
            last_cycle_data['predicted_rul'],
            bins=rul_bins,
            labels=rul_labels
        ).value_counts().sort_index()
        
        # Health distribution
        health_dist = last_cycle_data['health_status'].value_counts()
        
        # Health percentage distribution (for analytics chart)
        health_pct_bins = [0, 20, 40, 60, 80, 100]
        health_pct_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        health_pct_dist = pd.cut(
            last_cycle_data['health_percentage'],
            bins=health_pct_bins,
            labels=health_pct_labels
        ).value_counts().sort_index()
        
        # Fleet trend (average health and RUL over cycles)
        fleet_trend = test_data.groupby('cycle').agg({
            'health_percentage': 'mean',
            'predicted_rul': 'mean'
        }).reset_index()
        
        # Sample every 10 cycles for performance
        fleet_trend = fleet_trend[fleet_trend['cycle'] % 10 == 0]
        
        # Model comparison (static data for demo)
        model_comparison = {
            'metrics': ['R² Score', 'RMSE', 'MAE'],
            'baseline': [0.85, 16.0, 12.5],
            'advanced': [0.93, 5.5, 4.2]
        }
        
        # Sensor importance (mock data - in production this would come from model.feature_importances_)
        sensor_importance = {
            'sensor_11': 0.15,
            'sensor_14': 0.12,
            'sensor_4': 0.10,
            'sensor_2': 0.08,
            'sensor_7': 0.07,
            'sensor_21': 0.06,
            'sensor_15': 0.05,
            'sensor_3': 0.04
        }
        
        return {
            'rul_distribution': rul_dist.to_dict(),
            'health_distribution': health_dist.to_dict(),
            'health_percentage_distribution': health_pct_dist.to_dict(),
            'fleet_trend': {
                'cycles': fleet_trend['cycle'].tolist(),
                'avg_health': fleet_trend['health_percentage'].tolist(),
                'avg_rul': fleet_trend['predicted_rul'].tolist()
            },
            'model_comparison': model_comparison,
            'sensor_importance': sensor_importance
        }
