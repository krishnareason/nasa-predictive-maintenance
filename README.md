# 🚀 NASA Turbofan Predictive Maintenance API

[![Frontend](https://img.shields.io/badge/Frontend-React.js-61DAFB?logo=react&logoColor=black)](#)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](#)
[![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20Scikit--Learn-F37626?logo=scikit-learn&logoColor=white)](#)
[![Deployment](https://img.shields.io/badge/Deployment-Vercel%20%7C%20Render-black?logo=vercel&logoColor=white)](#)

**Live Application:** [NASA Predictive Maintenance UI](https://nasa-predictive-maintenance.vercel.app/)  
**Live API Documentation:** [FastAPI Swagger UI](https://nasa-predictive-maintenance-project.onrender.com/docs)

## 📌 Project Architecture & Overview
An end-to-end full-stack machine learning pipeline designed to predict the **Remaining Useful Life (RUL)** of turbofan engines using simulated NASA telemetry data. The system utilizes a decoupled microservice architecture: a high-performance **FastAPI** backend for real-time model inference, integrated with a responsive, dark-mode **React** dashboard.

The application successfully handles multiple engine degradation datasets, ranging from standard operational conditions (FD001) to highly complex, multi-regime flight environments (FD004).

## 🧠 Machine Learning Methodology
The inference engine does not rely on simple linear regression. It implements a robust, multi-stage data processing and predictive pipeline:

* **Regime Clustering (K-Means):** For complex datasets (FD004), the operational settings (altitude, Mach number, throttle resolver angle) vary wildly. A K-Means clustering model (`n_clusters=6`) categorizes the live telemetry into distinct flight regimes before prediction.
* **Contextual Normalization:** Sensor readings are scaled using `StandardScaler` based strictly on their corresponding flight regime, preventing data leakage and skewed magnitude errors.
* **XGBoost Regression:** The core predictive model utilizes an optimized Gradient Boosting Regressor (`XGBRegressor`) to capture complex, non-linear degradation patterns across 21 distinct sensor channels.

## ⚙️ Technical Stack
* **Frontend:** React.js, Recharts (Dynamic Visualization), Lucide React (Icons), Axios.
* **Backend:** Python 3.11, FastAPI, Uvicorn, Pydantic (Strict Type Validation).
* **AI/ML:** XGBoost, Scikit-Learn, Pandas, Numpy, Joblib.
* **Infrastructure:** Vercel (Frontend Hosting), Render (Backend API Hosting), Git/GitHub.

---

## 💻 Local Development Setup

If you wish to run the full decoupled architecture on your local machine, follow these steps:

```
### 1. Clone the Repository

git clone [https://github.com/krishnareason/nasa-predictive-maintenance.git](https://github.com/krishnareason/nasa-predictive-maintenance.git)
cd nasa-predictive-maintenance

### 2. Backend Setup (FastAPI + Inference Engine)

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start the Uvicorn server
python -m uvicorn api.main:app --reload --port 8000

The API will be available at http://localhost:8000

### 3. Frontend Setup (React Dashboard)

Open a new terminal window and navigate to the frontend directory:
cd frontend

# Install Node dependencies
npm install

# Start the development server
npm start

The UI will be available at http://localhost:3000

### 📈 Future Enhancements

> Implementation of an Agentic AI layer to autonomously suggest maintenance schedules based on the RUL output.
> Time-series sequence modeling using LSTMs for historical degradation tracking.
```

# 👨‍💻 Developed By
Krishna Srivastava



