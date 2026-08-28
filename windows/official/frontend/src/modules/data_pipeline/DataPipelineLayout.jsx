import React from 'react';
import { NavLink, Outlet, Navigate, useLocation } from 'react-router-dom';

export default function DataPipelineLayout() {
    const location = useLocation();

    // Khai báo danh sách sub-modules
    const subModules = [
        { path: 'upload', label: '1. Ingestion & Upload', desc: 'Chargement des fichiers' },
        { path: 'initial-profiling', label: '2. Initial Profiling', desc: 'Profilage initial des données' },
        { path: 'data-analysis', label: '3. Data Analysis', desc: 'Analyse globale des données' },
        { path: 'single-column', label: '4. Single Column Analysis', desc: 'Analyse détaillée par colonne' },
        { path: 'multi-column', label: '5. Multi Column Analysis', desc: 'Analyse multi colonne' },
        { path: 'timeseries-analysis', label: '6. Time Series Analysis', desc: 'Analyse des series temporelles' },
        { path: 'missing-value', label: '7. Missing Value Imputation', desc: 'Impute des valeurs manquantes' },
        { path: 'outliers', label: '7. Outliers Imputation', desc: 'Impute des valeurs abberantes' },
    ];

    return (
        <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Header Module */}
            <div style={{ marginBottom: '24px' }}>
                <h1 style={{ fontSize: '24px', fontWeight: 'bold', color: '#0f172a' }}>
                    Data Pipeline Module
                </h1>
                <p style={{ color: '#64748b', fontSize: '14px' }}>
                    Gestion, profilage et prétraitement des données d'entrée.
                </p>
            </div>

            {/* Thanh Tab / Sub-Navigation */}
            <div style={{ display: 'flex', gap: '12px', borderBottom: '2px solid #e2e8f0', marginBottom: '24px' }}>
                {subModules.map((tab) => (
                    <NavLink
                        key={tab.path}
                        to={tab.path}
                        style={({ isActive }) => ({
                            padding: '12px 20px',
                            textDecoration: 'none',
                            fontWeight: isActive ? '600' : '500',
                            color: isActive ? '#2563eb' : '#64748b',
                            borderBottom: isActive ? '3px solid #2563eb' : '3px solid transparent',
                            marginBottom: '-2px',
                            transition: 'all 0.2s',
                        })}
                    >
                        {tab.label}
                    </NavLink>
                ))}
            </div>

            {/* Nơi hiển thị các component con tương ứng với URL */}
            <div style={{ backgroundColor: '#ffffff', borderRadius: '8px', minHeight: '500px' }}>
                <Outlet />
            </div>
        </div>
    );
}