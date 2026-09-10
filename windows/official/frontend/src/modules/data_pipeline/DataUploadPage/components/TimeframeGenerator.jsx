import React, { useState } from 'react';
import axios from 'axios';
import { CheckCircle, AlertCircle, Loader2, Database, RotateCcw } from 'lucide-react';

export default function TimeframeGenerator({ onTimeframeGenerated }) {
    const defaultTimeframe = {
        start_date: '2022-01-01 00:00:00',
        end_date: '2026-07-30 23:00:00',
        attractions: 'H03, H07',
    };

    const [timeframe, setTimeframe] = useState(defaultTimeframe);
    const [loading, setLoading] = useState(false);
    const [response, setResponse] = useState(null);
    const [error, setError] = useState(null);

    const handleResetAll = () => {
        setTimeframe(defaultTimeframe);
        setResponse(null);
        setError(null);
    };

    const handleGenerateTimeframe = async () => {
        setLoading(true);
        setError(null);
        setResponse(null);

        const attractionsArray = timeframe.attractions
            .split(',')
            .map((item) => item.trim())
            .filter((item) => item.length > 0);

        try {
            const res = await axios.post('http://localhost:8000/etl/generate-fact-shells', {
                start_date: timeframe.start_date,
                end_date: timeframe.end_date,
                attractions: attractionsArray,
            });
            setResponse(res.data);
            if (onTimeframeGenerated && res.data?.table_name) {
                onTimeframeGenerated(res.data.table_name);
            }
        } catch (err) {
            setError(err.response?.data?.detail || "Erreur lors de l'initialisation de la période !");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ marginBottom: '40px', paddingBottom: '24px', borderBottom: '2px solid #e2e8f0' }}>
            {/* Header Area (Đồng bộ format với FileUploadModule) */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <h2 style={{ color: '#1e293b', margin: 0, fontSize: '20px', fontWeight: '700' }}>
                    Module 2 : Générateur de Période & Coques de Faits
                </h2>
                <button
                    onClick={handleResetAll}
                    disabled={loading}
                    style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '6px',
                        background: 'transparent',
                        border: '1px solid #cbd5e1',
                        borderRadius: '6px',
                        padding: '6px 12px',
                        fontSize: '13px',
                        color: '#64748b',
                        cursor: loading ? 'not-allowed' : 'pointer',
                    }}
                >
                    <RotateCcw size={14} /> Réinitialiser
                </button>
            </div>

            <p style={{ color: '#64748b', marginBottom: '24px', fontSize: '14px', lineHeight: '1.5' }}>
                Générez au préalable les lignes vides (visitor_count = NULL) à partir de la table dim_temps pour synchroniser la période avant l'arrivée des données réelles (ex : 2022 - 2024).
            </p>

            {/* Input Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '16px' }}>
                <div>
                    <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#334155', marginBottom: '6px' }}>
                        Date de début
                    </label>
                    <input
                        type="text"
                        value={timeframe.start_date}
                        onChange={(e) => setTimeframe({ ...timeframe, start_date: e.target.value })}
                        placeholder="2022-01-01 00:00:00"
                        style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '14px', fontFamily: 'monospace' }}
                    />
                </div>
                <div>
                    <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#334155', marginBottom: '6px' }}>
                        Date de fin
                    </label>
                    <input
                        type="text"
                        value={timeframe.end_date}
                        onChange={(e) => setTimeframe({ ...timeframe, end_date: e.target.value })}
                        placeholder="2026-07-30 23:00:00"
                        style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '14px', fontFamily: 'monospace' }}
                    />
                </div>
            </div>

            <div style={{ marginBottom: '24px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#334155', marginBottom: '6px' }}>
                    Codes des attractions (séparés par une virgule)
                </label>
                <input
                    type="text"
                    value={timeframe.attractions}
                    onChange={(e) => setTimeframe({ ...timeframe, attractions: e.target.value })}
                    placeholder="H03, H07"
                    style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '14px', fontFamily: 'monospace' }}
                />
            </div>

            {/* Submit Button (Đồng bộ format với FileUploadModule) */}
            <button
                onClick={handleGenerateTimeframe}
                disabled={loading}
                style={{
                    width: '100%',
                    padding: '14px',
                    backgroundColor: loading ? '#94a3b8' : '#059669',
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '16px',
                    fontWeight: '600',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '10px',
                }}
            >
                {loading ? (
                    <>
                        <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} /> 
                        Génération de la grille de données en cours...
                    </>
                ) : (
                    <>
                        <Database size={20} /> Initialiser les Fact Shells (Période)
                    </>
                )}
            </button>

            {/* Alert Boxes (Đồng bộ format với FileUploadModule) */}
            {response && (
                <div style={{ marginTop: '20px', padding: '16px', backgroundColor: '#dcfce7', borderRadius: '8px', color: '#15803d', border: '1px solid #86efac' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <CheckCircle size={22} style={{ flexShrink: 0 }} />
                        <strong style={{ fontSize: '15px' }}>{response.message}</strong>
                    </div>
                </div>
            )}

            {error && (
                <div style={{ marginTop: '20px', padding: '16px', backgroundColor: '#fee2e2', borderRadius: '8px', color: '#b91c1c', border: '1px solid #fca5a5', display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <AlertCircle size={22} style={{ flexShrink: 0 }} />
                    <div style={{ fontSize: '14px' }}>
                        <strong>Échec :</strong> {error}
                    </div>
                </div>
            )}

            {/* CSS Animation */}
            <style>{`
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    );
}