# AeroGuard: Advanced Jet Propulsion Monitoring System

**IIT Kharagpur Hackathon (Kshitij) - AI Track**

## 🎯 Problem Statement

Build a production-quality **Predictive Maintenance System** that predicts the Remaining Useful Life (RUL) of jet engines using the NASA CMAPSS FD001 dataset. The goal is to move from reactive repairs to pro-active maintenance, minimizing downtime and ensuring safety.

## 🏆 Solution Overview

**AeroGuard** is a next-generation monitoring dashboard that combines **XGBoost** machine learning with a **Mission Control** interface. It transforms raw sensor telemetry into actionable intelligence.

### 🧠 Core Technology

1.  **Balanced XGBoost Model**:
    *   Trained on advanced feature engineering (Rolling Averages, EMA, Differencing).
    *   **R² Score**: ~0.93 (High reliability).
    *   **RMSE**: ~16 cycles (Precise prediction).
2.  **Health Score Algorithm**:
    *   Converts RUL (cycles) into an intuitive **0-100% Health Score**.
    *   **Logic**: `(Current RUL / Max RUL) * 100`, clamped for stability.
3.  **Modern Tech Stack**:
    *   **Backend**: Flask (Python) for fast, lightweight API serving.
    *   **Frontend**: Native HTML/JS with **Chart.js** for high-performance rendering.
    *   **Design**: "Deep Black" Mission Control aesthetic with glass-morphism.

## 📂 Project Structure

```
hackathon_project/
│
├── data/                          # NASA CMAPSS Dataset
│   ├── train_FD001.txt
│   └── test_FD001.txt
│
├── models/                        # Trained Models
│   ├── balanced_xgboost_model.pkl # optimized model
│   └── balanced_scaler.pkl        # dedicated scaler
│
├── templates/                     # Frontend Assets
│   └── index.html                 # Single-page Dashboard Application
│
├── utils/                         # Core Logic Modules
│   ├── data_loader.py             # Raw data parsing
│   ├── health_score.py            # Diagnostic algos
│   └── model.py                   # Model definition
│
├── train_balanced.py              # Advanced training script
├── web_app.py                     # Flask Server Entry Point
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure the NASA CMAPSS data files (`train_FD001.txt`, `test_FD001.txt`) are present in the `data/` directory.

### 3. Launch "AeroGuard" Dashboard

Start the Flask server:

```bash
python web_app.py
```

Open your browser to the local address (usually `http://127.0.0.1:5000`).

### 4. (Optional) Retrain Model

If you want to regenerate the model with new parameters:

```bash
python train_balanced.py
```

## 🖥️ Dashboard Features

*   **Fleet Overview**: Real-time grid view of all propulsion units with status badges.
*   **Deep Dive Analysis**: Click any engine to view:
    *   **Trajectory**: Historical health degradation curve.
    *   **Telemetry Frame**: Live sensor readings (Temps, Pressures, RPMs).
    *   **AI Recommendation**: Actionable advice (e.g., "Schedule maintenance in 12 cycles").
*   **Mission Control UI**:
    *   Dark Mode optimization for reduced eye strain in control rooms.
    *   High-contrast alerts using standard aviation color codes (Green/Amber/Red).

## 🧠 Model Innovation: "Balanced Training"

We found that standard models overfit to "healthy" engines because engines spend most of their life in a healthy state.

**Our Approach:**
1.  **Feature Engineering**: We generated 100+ features including Rolling Mean (w=10, 20), Exponential Moving Averages, and Sensor Deltas.
2.  **Sample Balancing**: We explicitly weighted the "failure" samples higher to teach the model to prioritize detecting faults over simply predicting "normal".

## 📊 Performance Metrics

| Metric | Performance | What it means |
|--------|-------------|---------------|
| **Validation RMSE** | **16.5 cycles** | On average, our prediction is within 16 cycles of actual failure. |
| **Validation R²** | **0.93** | The model explains 93% of the engine degradation variance. |
| **Inference Time** | **<15ms** | Real-time prediction suitable for high-frequency updates. |

## 👥 Authors

Built with ❤️ for **IIT Kharagpur Kshitij 2026**.

## 📄 License

MIT License. Open for innovation.
