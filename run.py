"""
AeroGuard - Jet Engine Predictive Maintenance System
=====================================================
Main Flask application entry point.

Author: Team TriBits
"""

import logging
from flask import Flask, render_template, jsonify
from pathlib import Path

from config import FLASK_CONFIG, LOGGING_CONFIG
from services.prediction_service import PredictionService
from services.analytics_service import AnalyticsService

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format']
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.update(FLASK_CONFIG)

# Initialize services
prediction_service = None
analytics_service = None


def initialize_services():
    """Initialize application services."""
    global prediction_service, analytics_service
    
    try:
        logger.info("Initializing AeroGuard services...")
        prediction_service = PredictionService()
        analytics_service = AnalyticsService(prediction_service)
        logger.info("✓ Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


# Routes
@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/api/overview')
def get_overview():
    """Get fleet overview statistics."""
    try:
        overview = analytics_service.get_overview()
        return jsonify(overview)
    except Exception as e:
        logger.error(f"Error getting overview: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/engines')
def get_engines():
    """Get list of all engines with current status."""
    try:
        engines = prediction_service.get_all_engines()
        return jsonify(engines)
    except Exception as e:
        logger.error(f"Error getting engines: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/engine/<int:engine_id>')
def get_engine_details(engine_id):
    """Get detailed information for a specific engine."""
    try:
        details = prediction_service.get_engine_details(engine_id)
        if details is None:
            return jsonify({'error': 'Engine not found'}), 404
        return jsonify(details)
    except Exception as e:
        logger.error(f"Error getting engine {engine_id} details: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics')
def get_analytics():
    """Get analytics data for charts."""
    try:
        analytics = analytics_service.get_analytics()
        return jsonify(analytics)
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    """Main application entry point."""
    print("=" * 60)
    print("🚀 AeroGuard: Predictive Maintenance System")
    print("=" * 60)
    
    try:
        initialize_services()
        
        print(f"\n✓ Dashboard ready!")
        print(f"🌐 Open your browser: http://localhost:{FLASK_CONFIG['PORT']}")
        print("=" * 60)
        
        app.run(
            host=FLASK_CONFIG['HOST'],
            port=FLASK_CONFIG['PORT'],
            debug=FLASK_CONFIG['DEBUG'],
            threaded=FLASK_CONFIG['THREADED']
        )
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise


if __name__ == '__main__':
    main()
