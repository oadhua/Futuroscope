import React, { useState } from 'react';
import {
  Database,
  BrainCircuit,
  Wind,
  UploadCloud,
  BarChart3,
  Columns,
  Cpu,
  Activity,
  Sparkles,
  Settings,
  ChevronRight,
  Clock,
  Zap,
  PlaySquare
} from 'lucide-react';

// Imports Data Pipeline
import DataUploadPage from './modules/data_pipeline/DataUploadPage/DataUploadPage';
import InitialProfiling from './modules/data_pipeline/InitialProfiling/InitialProfiling';
import DataAnalysis from './modules/data_pipeline/DataAnalysis/DataAnalysis';
import SingleColumn from './modules/data_pipeline/SingleColumn/SingleColumn';
import MultiColumn from './modules/data_pipeline/MultiColumn/MultiColumn';
import TimeSeriesAnalysis from './modules/data_pipeline/TimeSeriesAnalysis/TimeSeriesAnalysis';
import MissingValue from './modules/data_pipeline/MissingValue/MissingValue';
import Outliers from './modules/data_pipeline/Outliers/Outliers';
import Scaling from './modules/data_pipeline/Scaling/Scaling';
import Encoding from './modules/data_pipeline/Encoding/Encode';
import Duplicate from './modules/data_pipeline/Duplicated/Duplicated';
import TimeSeriesTrans from './modules/data_pipeline/TimeSeriesTrans/TimeSeriesTrans';
import Trace from './modules/data_pipeline/Trace/Trace';
import FeatureSelection from './modules/data_pipeline/FeatureSelection/FeatureSelection';

// Import ML Module
import MLTrainingDashboard from './modules/ml_forecasting/Training/Training';
import MLInference from './modules/ml_forecasting/MLInference/MLInference';
import FutureForecast from './modules/ml_forecasting/FutureForecast/FutureForecast';

export default function App() {
  const [activeTab, setActiveTab] = useState('pipeline');
  const [activeSubTab, setActiveSubTab] = useState('upload');

  const menuConfig = {
    pipeline: {
      title: 'Data Pipeline',
      icon: <Database size={18} />,
      subTabs: [
        { id: 'upload', label: '1.1 Data Upload & ETL', icon: <UploadCloud size={16} /> },
        { id: 'initialProfiling', label: '1.2 Initial Profiling', icon: <BarChart3 size={16} /> },
        { id: 'dataAnalysis', label: '1.3 Data Analysis', icon: <BarChart3 size={16} /> },
        { id: 'singleColumn', label: '1.4 Single Column Analysis', icon: <Columns size={16} /> },
        { id: 'multiColumn', label: '1.5 Multi Column Analysis', icon: <Columns size={16} /> },
        { id: 'timeseriesAnalysis', label: '1.6 Time Series Analysis', icon: <Clock size={16} /> },
        { id: 'missingValue', label: '1.7 Missing Value Imputation', icon: <Activity size={16} /> },
        { id: 'outliers', label: '1.8 Outliers Imputation', icon: <Activity size={16} /> },
        { id: 'scaling', label: '1.9 Scaling', icon: <Activity size={16} /> },
        { id: 'encoding', label: '1.10 Encoding', icon: <Activity size={16} /> },
        { id: 'duplicated', label: '1.11 Duplicated Data Removal', icon: <Activity size={16} /> },
        { id: 'timeSeriesTrans', label: '1.12 Time Series Transformation', icon: <Activity size={16} /> },
        { id: 'trace', label: '1.13 Traceability', icon: <Activity size={16} /> },
        { id: 'featureSelection', label: '1.14 Feature Selection', icon: <Activity size={16} /> }
      ]
    },
    ml: {
      title: 'ML Forecasting',
      icon: <BrainCircuit size={18} />,
      subTabs: [
        { id: 'training', label: '2.1 Model Training & Studio', icon: <PlaySquare size={16} /> },
        { id: 'visitor', label: '2.2 Evaluation & Comparison', icon: <Activity size={16} /> },
        { id: 'energy', label: '2.3 Prediction', icon: <Zap size={16} /> },
      ]
    },
    // hvac: {
    //   title: 'HVAC Optimization',
    //   icon: <Wind size={18} />,
    //   subTabs: [
    //     { id: 'control', label: '3.1 Thermal Model & Control', icon: <Sparkles size={16} /> },
    //     { id: 'settings', label: '3.2 Optimization Rules', icon: <Settings size={16} /> },
    //   ]
    // }
  };

  const handleMainTabChange = (tabKey) => {
    setActiveTab(tabKey);
    setActiveSubTab(menuConfig[tabKey].subTabs[0].id);
  };

  const renderContent = () => {
    if (activeTab === 'pipeline') {
      if (activeSubTab === 'upload') return <DataUploadPage />;
      if (activeSubTab === 'initialProfiling') return <InitialProfiling />;
      if (activeSubTab === 'dataAnalysis') return <DataAnalysis />;
      if (activeSubTab === 'singleColumn') return <SingleColumn />;
      if (activeSubTab === 'multiColumn') return <MultiColumn />;
      if (activeSubTab === 'timeseriesAnalysis') return <TimeSeriesAnalysis />;
      if (activeSubTab === 'missingValue') return <MissingValue />;
      if (activeSubTab === 'outliers') return <Outliers />;
      if (activeSubTab === 'scaling') return <Scaling />;
      if (activeSubTab === 'encoding') return <Encoding />;
      if (activeSubTab === 'duplicated') return <Duplicate />;
      if (activeSubTab === 'timeSeriesTrans') return <TimeSeriesTrans />;
      if (activeSubTab === 'trace') return <Trace />;
      if (activeSubTab === 'featureSelection') return <FeatureSelection />;
    }

    if (activeTab === 'ml') {
      if (activeSubTab === 'training') return <MLTrainingDashboard />;
      if (activeSubTab === 'visitor') return <MLInference />;
      if (activeSubTab === 'energy') return <FutureForecast />;
    }

    // if (activeTab === 'hvac') {
    //   if (activeSubTab === 'control') return <PlaceholderContent title="HVAC Smart Control & Simulation" />;
    //   if (activeSubTab === 'settings') return <PlaceholderContent title="HVAC Energy Savings Rules & Constraints" />;
    // }

    return null;
  };

  return (
    <div style={{ display: 'flex', width: '100vw', minHeight: '100vh', fontFamily: 'Inter, system-ui, sans-serif', backgroundColor: '#f8fafc', overflowX: 'hidden' }}>

      {/* 1. LEFT SIDEBAR */}
      <aside style={{
        width: '250px',
        backgroundColor: '#0f172a',
        color: '#f8fafc',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        borderRight: '1px solid #1e293b'
      }}>
        {/* Brand Header */}
        <div style={{ padding: '20px 16px', borderBottom: '1px solid #1e293b' }}>
          <h1 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#ffffff', letterSpacing: '-0.3px' }}>
            Futuroscope Energy
          </h1>
          <p style={{ margin: '3px 0 0 0', fontSize: '11px', color: '#94a3b8' }}>
            LIAS Lab & Futuroscope Platform
          </p>
        </div>

        {/* Menu Navigation */}
        <nav style={{ flex: 1, padding: '12px 8px', overflowY: 'auto' }}>
          {Object.keys(menuConfig).map((key) => {
            const item = menuConfig[key];
            const isMainActive = activeTab === key;

            return (
              <div key={key} style={{ marginBottom: '6px' }}>
                <button
                  onClick={() => handleMainTabChange(key)}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 10px',
                    borderRadius: '8px',
                    border: 'none',
                    backgroundColor: isMainActive ? '#1e293b' : 'transparent',
                    color: isMainActive ? '#38bdf8' : '#94a3b8',
                    fontWeight: isMainActive ? '600' : '500',
                    fontSize: '13px',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {item.icon}
                    <span>{item.title}</span>
                  </div>
                  <ChevronRight
                    size={14}
                    style={{
                      transform: isMainActive ? 'rotate(90deg)' : 'rotate(0deg)',
                      transition: 'transform 0.2s'
                    }}
                  />
                </button>

                {isMainActive && (
                  <div style={{ marginTop: '2px', paddingLeft: '12px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                    {item.subTabs.map((sub) => {
                      const isSubActive = activeSubTab === sub.id;
                      return (
                        <button
                          key={sub.id}
                          onClick={() => setActiveSubTab(sub.id)}
                          style={{
                            width: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            padding: '7px 10px',
                            borderRadius: '6px',
                            border: 'none',
                            backgroundColor: isSubActive ? '#2563eb' : 'transparent',
                            color: isSubActive ? '#ffffff' : '#94a3b8',
                            fontSize: '12px',
                            fontWeight: isSubActive ? '600' : '400',
                            cursor: 'pointer',
                            textAlign: 'left',
                            transition: 'all 0.15s ease'
                          }}
                        >
                          {sub.icon}
                          <span>{sub.label}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </nav>

        <div style={{ padding: '12px 16px', borderTop: '1px solid #1e293b', fontSize: '11px', color: '#64748b' }}>
          v2.4.0 • Academic & Industrial
        </div>
      </aside>

      {/* 2. MAIN CONTENT AREA */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', width: 'calc(100vw - 250px)', overflow: 'hidden' }}>

        {/* Breadcrumb Top Header */}
        <header style={{
          height: '48px',
          backgroundColor: '#ffffff',
          borderBottom: '1px solid #e2e8f0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 20px',
          flexShrink: 0
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: '#64748b' }}>
            <span>{menuConfig[activeTab].title}</span>
            <span>/</span>
            <span style={{ fontWeight: '600', color: '#0f172a' }}>
              {menuConfig[activeTab].subTabs.find(s => s.id === activeSubTab)?.label}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ height: '8px', width: '8px', borderRadius: '50%', backgroundColor: '#10b981' }}></span>
            <span style={{ fontSize: '12px', color: '#64748b', fontWeight: '500' }}>System Online</span>
          </div>
        </header>

        {/* Nội dung trang */}
        <div style={{ flex: 1, padding: '16px', overflowY: 'auto' }}>
          {renderContent()}
        </div>
      </main>
    </div>
  );
}

function PlaceholderContent({ title }) {
  return (
    <div style={{
      backgroundColor: '#ffffff',
      borderRadius: '12px',
      border: '1px solid #e2e8f0',
      padding: '48px',
      textAlign: 'center',
      marginTop: '12px'
    }}>
      <div style={{ width: '48px', height: '48px', borderRadius: '50%', backgroundColor: '#eff6ff', color: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px auto' }}>
        <Sparkles size={24} />
      </div>
      <h3 style={{ margin: '0 0 8px 0', fontSize: '18px', fontWeight: 'bold', color: '#0f172a' }}>{title}</h3>
      <p style={{ margin: 0, fontSize: '14px', color: '#64748b' }}>
        Module này đang được phát triển. Vui lòng quay lại sau!
      </p>
    </div>
  );
}