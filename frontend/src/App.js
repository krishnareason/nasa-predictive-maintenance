import React, { useState } from 'react';
import axios from 'axios';
import { Activity, Server, Settings } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './App.css';

const fd001Data = {
  time_cycle: 180, op_setting_1: -0.0014, op_setting_2: -0.0003,
  sensor_2: 643.31, sensor_3: 1593.61, sensor_4: 1414.86, sensor_5: 14.62,
  sensor_6: 21.61, sensor_7: 551.42, sensor_8: 2388.12, sensor_9: 9054.86,
  sensor_11: 48.31, sensor_12: 520.01, sensor_13: 2388.15, sensor_14: 8144.10,
  sensor_15: 8.522, sensor_16: 0.03, sensor_17: 396.0, sensor_20: 38.42, sensor_21: 23.13
};

const fd004Data = {
  time_cycle: 250, op_setting_1: 42.0077, op_setting_2: 0.8400, op_setting_3: 100.0,
  sensor_1: 445.00, sensor_2: 549.68, sensor_3: 1345.15, sensor_4: 1123.55, sensor_5: 3.91,
  sensor_6: 5.71, sensor_7: 138.82, sensor_8: 2211.83, sensor_9: 8312.44, sensor_10: 1.02,
  sensor_11: 41.98, sensor_12: 130.55, sensor_13: 2387.95, sensor_14: 8084.10, sensor_15: 9.33,
  sensor_16: 0.02, sensor_17: 330.0, sensor_18: 2212.0, sensor_19: 100.00, sensor_20: 10.55, sensor_21: 6.33
};

const historicalTelemetry = [
  { cycle: 130, sensor_14: 8120.40 }, { cycle: 140, sensor_14: 8125.10 },
  { cycle: 150, sensor_14: 8130.80 }, { cycle: 160, sensor_14: 8135.50 },
  { cycle: 170, sensor_14: 8140.20 }, { cycle: 180, sensor_14: 8144.10 },
];

function App() {
  const [activeTab, setActiveTab] = useState('FD001');
  const [formData, setFormData] = useState(fd001Data);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const BASE_API_URL = "https://nasa-predictive-maintenance-project.onrender.com";
  //const BASE_API_URL = "http://localhost:8000";
  const switchEngine = (engineType) => {
    setActiveTab(engineType);
    setResult(null);
    setFormData(engineType === 'FD001' ? fd001Data : fd004Data);
  };

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: parseFloat(e.target.value) || 0 });
  };

  const handlePredict = async () => {
    setLoading(true);
    const endpoint = activeTab === 'FD001' ? '/predict' : '/predict/fd004';
    
    try {
      const response = await axios.post(`${BASE_API_URL}${endpoint}`, formData);
      setResult(response.data);
    } catch (error) {
      console.error("API Error:", error);
      alert("Error connecting to cloud. Check if Render API is live.");
    }
    setLoading(false);
  };

  return (
    <div className="dashboard-container">
      <div className="header">
        <Activity color="#3b82f6" size={32} />
        <h1>NASA Fleet Monitoring AI</h1>
      </div>
      <div className="tab-container">
        <button 
          className={`tab-btn ${activeTab === 'FD001' ? 'active' : ''}`}
          onClick={() => switchEngine('FD001')}
        >
          Engine #1 (Standard conditions)
        </button>
        <button 
          className={`tab-btn ${activeTab === 'FD004' ? 'active' : ''}`}
          onClick={() => switchEngine('FD004')}
        >
          Engine #2 (Multi-Regime/Complex)
        </button>
      </div>

      <div className="grid-layout">
        <div className="panel">
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
            {activeTab === 'FD001' ? <Server size={20} color="#94a3b8" /> : <Settings size={20} color="#f59e0b" />}
            <h3 style={{ marginLeft: '8px' }}>
              Telemetry Feed ({activeTab === 'FD001' ? 'FD001 Base' : 'FD004 Advanced'})
            </h3>
          </div>
          
          <div className="form-grid" style={{ maxHeight: '400px', overflowY: 'auto', paddingRight: '10px' }}>
            {Object.keys(formData).map((key) => (
              <div className="input-group" key={key}>
                <label>{key.replace('_', ' ').toUpperCase()}</label>
                <input
                  type="number"
                  name={key}
                  value={formData[key]}
                  onChange={handleInputChange}
                  step="any"
                />
              </div>
            ))}
          </div>

          <button className="btn-predict" onClick={handlePredict} disabled={loading}>
            {loading ? "Routing to AI Cluster..." : `Run Inference on ${activeTab}`}
          </button>
        </div>
        <div className="panel">
          <h3>ML Inference Engine</h3>
          <p style={{ color: '#94a3b8', marginBottom: '2rem' }}>
            {activeTab === 'FD001' 
              ? 'Model: XGBoost Regression | Status: Unsupervised' 
              : 'Model: K-Means Normalized XGBoost | Status: Multi-Regime'}
          </p>
          
          {!result ? (
            <div style={{ textAlign: 'center', padding: '4rem 0', color: '#64748b' }}>
              Awaiting telemetry data injection...
            </div>
          ) : (
            <>
              <div className={`result-card status-${result.health_status.toLowerCase()}`}>
                <h2 style={{ margin: 0, textTransform: 'uppercase' }}>{result.health_status}</h2>
                <div className="rul-number">{result.predicted_rul_cycles}</div>
                <p style={{ margin: 0, fontSize: '1.2rem' }}>Cycles Remaining</p>
                <p style={{ marginTop: '1rem', color: '#64748b', fontSize: '0.9rem' }}>{result.engine_type}</p>
              </div>

              {activeTab === 'FD001' && (
                <div style={{ marginTop: '3rem' }}>
                  <h4 style={{ color: '#f8fafc', marginBottom: '1rem' }}>Sensor 14 Degradation Trend</h4>
                  <div style={{ height: '200px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={historicalTelemetry}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="cycle" stroke="#94a3b8" />
                        <YAxis domain={['auto', 'auto']} stroke="#94a3b8" />
                        <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
                        <Line type="monotone" dataKey="sensor_14" stroke="#ef4444" strokeWidth={3} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;