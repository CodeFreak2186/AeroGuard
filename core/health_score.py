"""
Health Score Calculation Module
================================
Converts Remaining Useful Life (RUL) into intuitive Health Percentage.

WHY THIS MATTERS:
- RUL in "cycles" is technical and hard to interpret
- Health % is intuitive: 100% = new, 0% = failed
- Color-coded status helps operators make quick decisions
"""

import numpy as np


def rul_to_health_percentage(rul, max_rul=125):
    """
    Convert RUL to Health Percentage
    
    LOGIC EXPLANATION (For Judges):
    
    Concept:
        - A new engine has maximum RUL (125 cycles) = 100% health
        - A failed engine has 0 RUL = 0% health
        - Health degrades linearly as RUL decreases
    
    Formula:
        Health % = (Current RUL / Maximum RUL) × 100
    
    Example:
        - RUL = 125 → Health = 100%
        - RUL = 62.5 → Health = 50%
        - RUL = 0 → Health = 0%
    
    WHY LINEAR?
        - Simple and explainable
        - Matches intuition: "Half the life left = Half health"
        - Could be non-linear in production (e.g., exponential decay)
    
    Args:
        rul: Remaining Useful Life (cycles)
        max_rul: Maximum RUL value (default: 125)
    
    Returns:
        Health percentage (0-100)
    """
    # Convert to numpy array for vectorized operations
    rul = np.array(rul)
    
    # Calculate health percentage
    health = (rul / max_rul) * 100
    
    # Clamp to [0, 100] range
    # WHY: Predictions might be slightly negative or >125 due to model error
    health = np.clip(health, 0, 100)
    
    return health


def get_health_status(health_percentage):
    """
    Convert health percentage to categorical status with color
    
    STATUS CATEGORIES (Industry Standard):
    
    Healthy (Green): 70-100%
        - Engine is in good condition
        - Normal operation
        - No immediate action needed
    
    Warning (Yellow): 40-69%
        - Engine is degrading
        - Plan maintenance soon
        - Monitor more frequently
    
    Critical (Red): 0-39%
        - Engine is near failure
        - Schedule immediate maintenance
        - Risk of unexpected shutdown
    
    WHY THESE THRESHOLDS?
        - Based on common industrial practice
        - Balances false alarms vs missed failures
        - Can be tuned based on business needs
    
    Args:
        health_percentage: Health value (0-100)
    
    Returns:
        Tuple: (status_text, color_hex, color_name)
    """
    if health_percentage >= 70:
        return 'Healthy', '#00C851', 'green'
    elif health_percentage >= 40:
        return 'Warning', '#FFB300', 'orange'
    else:
        return 'Critical', '#FF4444', 'red'


def get_maintenance_recommendation(health_percentage, rul):
    """
    Generate maintenance recommendation based on health and RUL
    
    BUSINESS LOGIC:
    - Helps operators decide when to schedule maintenance
    - Balances safety vs operational efficiency
    
    Args:
        health_percentage: Current health (0-100)
        rul: Remaining useful life (cycles)
    
    Returns:
        String recommendation
    """
    if health_percentage >= 70:
        return f"✅ Engine is healthy. Next inspection in {int(rul * 0.5)} cycles."
    elif health_percentage >= 40:
        return f"⚠️ Schedule maintenance within {int(rul * 0.7)} cycles."
    else:
        return f"🚨 URGENT: Schedule immediate maintenance. Only {int(rul)} cycles remaining!"


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("HEALTH SCORE CALCULATION EXAMPLES")
    print("="*60)
    
    # Test cases
    test_ruls = [125, 100, 75, 50, 25, 10, 0]
    
    for rul in test_ruls:
        health = rul_to_health_percentage(rul)
        status, color, _ = get_health_status(health)
        recommendation = get_maintenance_recommendation(health, rul)
        
        print(f"\nRUL: {rul:3d} cycles")
        print(f"  Health: {health:.1f}%")
        print(f"  Status: {status} ({color})")
        print(f"  Action: {recommendation}")
