import React, { useState, useEffect } from 'react';
import {
    Activity, TrendingUp, Sliders, Filter,
    ChevronDown, Play, CheckCircle2, AlertCircle,
    BarChart2, Layers, Clock, Loader2
} from 'lucide-react';
import Plot from 'react-plotly.js';

const API_BASE_URL = 'http://localhost:8000/time-series';

export default function TimeSeriesAnalysis() {
    const [activeTab, setActiveTab] = useState('stationarity');

    // Filter States
    const [version, setVersion] = useState('v1');
    const [selectedAttraction, setSelectedAttraction] = useState('Toutes');
    const [selectedColumn, setSelectedColumn] = useState('');

    // Dynamic Options States
    const [availableAttractions, setAvailableAttractions] = useState([]);
    const [availableColumns, setAvailableColumns] = useState([]);
    const [loadingOptions, setLoadingOptions] = useState(true);

    // Analysis Execution States
    const [loading, setLoading] = useState(false);

    // Tab States
    const [statMethod, setStatMethod] = useState('adf');
    const [statResult, setStatResult] = useState(null);

    const [decompModel, setDecompModel] = useState('additive');
    const [decompPeriod, setDecompPeriod] = useState(24);
    const [decompResult, setDecompResult] = useState(null);

    const [plotType, setPlotType] = useState('acf');
    const [lags, setLags] = useState(25);
    const [acfResult, setAcfResult] = useState(null);

    // 1. Fetch Metadata Options theo Attraction được chọn
    useEffect(() => {
        const fetchMetaOptions = async () => {
            setLoadingOptions(true);
            try {
                let url = `${API_BASE_URL}/meta/options`;

                if (selectedAttraction && selectedAttraction !== 'Toutes') {
                    url += `?id_attraction=${selectedAttraction}`;
                } else {
                    url += `?id_attraction=all`;
                }

                const res = await fetch(url);
                if (res.ok) {
                    const data = await res.json();

                    // 1. Cập nhật danh sách Attractions (chỉ cập nhật nếu chưa có)
                    if (data.attractions && data.attractions.length > 0) {
                        const attractionsList = ['Toutes', ...data.attractions.filter(a => a !== 'Toutes')];
                        setAvailableAttractions(attractionsList);
                    }

                    // 2. Cập nhật danh sách Cột (Features) theo attraction vừa chọn
                    const cols = data.features || [];
                    setAvailableColumns(cols);

                    // Always auto-select the first feature of the newly selected attraction
                    if (cols.length > 0) {
                        setSelectedColumn(cols[0]); // Tự động chọn cột đầu tiên của attraction mới
                    } else {
                        setSelectedColumn('');
                    }
                }
            } catch (err) {
                console.warn("Erreur lors du chargement des métadonnées:", err);
            } finally {
                setLoadingOptions(false);
            }
        };

        fetchMetaOptions();
    }, [selectedAttraction]);

    // Đổi Attraction -> Reset ngay cột đang chọn để tránh lỗi khớp dữ liệu
    const handleAttractionChange = (e) => {
        const newAttraction = e.target.value;
        setSelectedAttraction(newAttraction);
        setSelectedColumn('');
    };

    // 2. Fetch Data Function
    const fetchData = async () => {
        if (!selectedColumn) return;

        setLoading(true);
        try {
            const params = new URLSearchParams({
                version,
                col_name: selectedColumn,
            });

            // Xử lý tham số id_attraction khi gửi lên Backend
            if (selectedAttraction === 'Toutes') {
                params.append('id_attraction', 'all'); // Bạn có thể sửa 'all' thành '' nếu Backend chấp nhận để trống
            } else if (selectedAttraction) {
                params.append('id_attraction', selectedAttraction);
            }

            if (activeTab === 'stationarity') {
                params.append('method', statMethod);
                const res = await fetch(`${API_BASE_URL}/stationarity?${params.toString()}`);
                const data = await res.json();
                setStatResult(data);
            } else if (activeTab === 'decomposition') {
                params.append('model_type', decompModel);
                params.append('period', decompPeriod.toString());
                const res = await fetch(`${API_BASE_URL}/decomposition?${params.toString()}`);
                const data = await res.json();
                setDecompResult(data);
            } else if (activeTab === 'acf-pacf') {
                params.append('plot_type', plotType);
                params.append('lags', lags.toString());
                const res = await fetch(`${API_BASE_URL}/acf-pacf?${params.toString()}`);
                const data = await res.json();
                setAcfResult(data);
            }
        } catch (err) {
            console.error('Erreur lors du chargement des données:', err);
        } finally {
            setLoading(false);
        }
    };

    // Tự động tính toán khi chuyển Tab
    useEffect(() => {
        if (!loadingOptions && selectedColumn) {
            fetchData();
        }
    }, [activeTab]);

    const styles = {
        wrapper: { minHeight: '100vh', backgroundColor: '#f8fafc', padding: '24px', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1e293b' },
        container: { maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' },
        card: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)', boxSizing: 'border-box' },
        headerCard: {
            backgroundColor: '#ffffff',
            borderRadius: '24px',
            border: '1px solid #e2e8f0',
            padding: '16px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '16px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
            minHeight: '80px',
            boxSizing: 'border-box',
            flexWrap: 'wrap'
        },
        selectBox: {
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            backgroundColor: '#f8fafc',
            border: '1px solid #e2e8f0',
            padding: '8px 14px',
            borderRadius: '16px',
            fontSize: '12px',
            minWidth: '180px',
            maxWidth: '240px',
            flexShrink: 0
        },
        select: {
            border: 'none',
            background: 'transparent',
            fontWeight: 'bold',
            color: '#0f172a',
            outline: 'none',
            cursor: 'pointer',
            appearance: 'none',
            width: '100%',
            textOverflow: 'ellipsis',
            overflow: 'hidden',
            whiteSpace: 'nowrap'
        },
        iconBg: (bg, color) => ({ backgroundColor: bg, color: color, padding: '10px', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }),
        tabBtn: (isActive) => ({
            padding: '8px 16px',
            borderRadius: '12px',
            fontSize: '13px',
            fontWeight: 'bold',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s',
            backgroundColor: isActive ? '#ffffff' : 'transparent',
            color: isActive ? '#2563eb' : '#64748b',
            boxShadow: isActive ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
        }),
        actionBtn: {
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            backgroundColor: '#2563eb',
            color: '#ffffff',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '16px',
            fontSize: '12px',
            fontWeight: 'bold',
            cursor: 'pointer',
            transition: 'all 0.2s',
            height: '38px',
            flexShrink: 0
        },
        radioBtn: (isSelected) => ({
            padding: '6px 14px',
            borderRadius: '12px',
            fontSize: '12px',
            fontWeight: 'bold',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s',
            backgroundColor: isSelected ? '#ffffff' : 'transparent',
            color: isSelected ? '#2563eb' : '#64748b',
            boxShadow: isSelected ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
        }),
        loadingContainer: {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '60px 20px',
            gap: '12px',
            color: '#64748b',
            fontSize: '13px',
            fontWeight: '500'
        }
    };

    return (
        <div style={styles.wrapper}>
            <style>
                {`
                    @keyframes spin {
                        from { transform: rotate(0deg); }
                        to { transform: rotate(360deg); }
                    }
                    .spinner {
                        animation: spin 1s linear infinite;
                    }
                `}
            </style>
            <div style={styles.container}>

                {/* 1. TOP HEADER & FILTER CARD */}
                <div style={styles.headerCard}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flex: '1 1 auto', minWidth: 0 }}>
                        <div style={styles.iconBg('#eff6ff', '#2563eb')}>
                            <Activity size={24} />
                        </div>
                        <div style={{ minWidth: 0, overflow: 'hidden' }}>
                            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.02em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                Profilage et Analyse des Séries Temporelles
                            </h1>
                            <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                Variable sélectionnée : <span style={{ color: '#2563eb' }}>{selectedColumn || 'Chargement...'}</span>
                            </p>
                        </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap', flexShrink: 0 }}>
                        {/* Dropdown Version */}
                        <div style={styles.selectBox}>
                            <Layers size={15} color="#94a3b8" style={{ flexShrink: 0 }} />
                            <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>Version :</span>
                            <select value={version} onChange={(e) => setVersion(e.target.value)} style={styles.select}>
                                <option value="v1">Version 1 (Par défaut)</option>
                                <option value="v2">Version 2</option>
                            </select>
                            <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                        </div>

                        {/* Dropdown Attraction */}
                        <div style={styles.selectBox}>
                            <Filter size={15} color="#94a3b8" style={{ flexShrink: 0 }} />
                            <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>Attraction :</span>
                            <select
                                value={selectedAttraction}
                                onChange={handleAttractionChange}
                                style={styles.select}
                            >
                                {availableAttractions.map(id => (
                                    <option key={id} value={id}>
                                        {id === 'Toutes' ? 'Toutes les attractions' : id}
                                    </option>
                                ))}
                            </select>
                            <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                        </div>

                        {/* Dropdown Colonne */}
                        <div style={styles.selectBox}>
                            <Sliders size={15} color="#94a3b8" style={{ flexShrink: 0 }} />
                            <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>Colonne :</span>
                            <select
                                value={selectedColumn}
                                onChange={(e) => setSelectedColumn(e.target.value)}
                                style={styles.select}
                                disabled={loadingOptions || availableColumns.length === 0}
                            >
                                {availableColumns.length > 0 ? (
                                    availableColumns.map(col => <option key={col} value={col}>{col}</option>)
                                ) : (
                                    <option value="">Aucune colonne</option>
                                )}
                            </select>
                            <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                        </div>

                        <button
                            onClick={fetchData}
                            disabled={loading || loadingOptions || !selectedColumn}
                            style={{
                                ...styles.actionBtn,
                                backgroundColor: (loading || !selectedColumn) ? '#93c5fd' : '#2563eb',
                                cursor: (loading || !selectedColumn) ? 'not-allowed' : 'pointer'
                            }}
                        >
                            {loading ? (
                                <>
                                    <Loader2 size={14} className="spinner" />
                                    <span>Calcul...</span>
                                </>
                            ) : (
                                <>
                                    <Play size={14} />
                                    <span>Exécuter</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {/* 2. TAB NAVIGATION */}
                <div style={{ display: 'flex', backgroundColor: '#f1f5f9', padding: '4px', borderRadius: '16px', gap: '4px', width: 'fit-content' }}>
                    <button onClick={() => setActiveTab('stationarity')} style={styles.tabBtn(activeTab === 'stationarity')}>
                        Test de Stationnarité
                    </button>
                    <button onClick={() => setActiveTab('decomposition')} style={styles.tabBtn(activeTab === 'decomposition')}>
                        Décomposition Temporelle
                    </button>
                    <button onClick={() => setActiveTab('acf-pacf')} style={styles.tabBtn(activeTab === 'acf-pacf')}>
                        Fonctions ACF / PACF
                    </button>
                </div>

                {/* 3. CONTENT CARDS */}
                {activeTab === 'stationarity' && (
                    <div style={styles.card}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <div style={styles.iconBg('#eff6ff', '#2563eb')}>
                                    <Activity size={20} />
                                </div>
                                <div>
                                    <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a' }}>Test de Stationnarité</h2>
                                    <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>Évaluation de la stationnarité via les tests ADF ou KPSS</p>
                                </div>
                            </div>

                            <div style={{ display: 'flex', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '4px', borderRadius: '16px', gap: '4px' }}>
                                <button onClick={() => setStatMethod('adf')} style={styles.radioBtn(statMethod === 'adf')}>Test ADF</button>
                                <button onClick={() => setStatMethod('kpss')} style={styles.radioBtn(statMethod === 'kpss')}>Test KPSS</button>
                            </div>
                        </div>

                        {loading ? (
                            <div style={styles.loadingContainer}>
                                <Loader2 size={32} color="#2563eb" className="spinner" />
                                <span>Chargement du test de stationnarité...</span>
                            </div>
                        ) : statResult ? (
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
                                <div style={{
                                    padding: '20px', borderRadius: '16px', border: '1px solid',
                                    backgroundColor: statResult.is_stationary ? '#f0fdf4' : '#fef2f2',
                                    borderColor: statResult.is_stationary ? '#bbf7d0' : '#fecaca',
                                    color: statResult.is_stationary ? '#166534' : '#991b1b',
                                    display: 'flex', flexDirection: 'column', justifyContent: 'center'
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                                        {statResult.is_stationary ? <CheckCircle2 size={20} /> : <AlertCircle size={20} />}
                                        <h3 style={{ fontSize: '16px', fontWeight: '800', margin: 0 }}>{statResult.message}</h3>
                                    </div>
                                    <p style={{ fontSize: '12px', margin: 0, opacity: 0.9 }}>
                                        Méthode : <strong>{statResult.method}</strong> | Colonne : <strong>{statResult.col_name}</strong>
                                    </p>
                                </div>

                                <div style={{ padding: '20px', backgroundColor: '#f8fafc', borderRadius: '16px', border: '1px solid #e2e8f0' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '8px' }}>
                                        <span style={{ color: '#64748b', fontWeight: '500' }}>Statistique du test :</span>
                                        <strong style={{ fontFamily: 'monospace', color: '#0f172a' }}>{statResult.test_statistic}</strong>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '12px' }}>
                                        <span style={{ color: '#64748b', fontWeight: '500' }}>p-valeur :</span>
                                        <strong style={{ fontFamily: 'monospace', color: '#0f172a' }}>{statResult.p_value}</strong>
                                    </div>
                                    <div style={{ paddingTop: '10px', borderTop: '1px solid #e2e8f0' }}>
                                        <span style={{ fontSize: '11px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', display: 'block', marginBottom: '6px' }}>Valeurs critiques :</span>
                                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                            {Object.entries(statResult.critical_values || {}).map(([k, v]) => (
                                                <span key={k} style={{ backgroundColor: '#ffffff', padding: '4px 8px', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px', fontFamily: 'monospace', color: '#334155', fontWeight: '600' }}>
                                                    {k}: {v}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div style={{ display: 'flex', justifyContent: 'center', padding: '40px', color: '#94a3b8', fontSize: '13px', fontStyle: 'italic' }}>
                                Aucune donnée de stationnarité disponible.
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'decomposition' && (
                    <div style={styles.card}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <div style={styles.iconBg('#f0fdf4', '#16a34a')}>
                                    <TrendingUp size={20} />
                                </div>
                                <div>
                                    <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a' }}>Décomposition de Série Temporelle</h2>
                                    <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>Séparation des composantes : Tendance, Saisonnalité et Résidus</p>
                                </div>
                            </div>

                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                                <div style={styles.selectBox}>
                                    <span style={{ color: '#64748b', fontWeight: '600', flexShrink: 0 }}>Modèle :</span>
                                    <select value={decompModel} onChange={(e) => setDecompModel(e.target.value)} style={styles.select}>
                                        <option value="additive">Additif (+)</option>
                                        <option value="multiplicative">Multiplicatif (*)</option>
                                    </select>
                                    <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                                </div>

                                <div style={{ ...styles.selectBox, minWidth: '130px', maxWidth: '150px' }}>
                                    <Clock size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                                    <span style={{ color: '#64748b', fontWeight: '600', flexShrink: 0 }}>Période :</span>
                                    <input
                                        type="number"
                                        value={decompPeriod}
                                        onChange={(e) => setDecompPeriod(Number(e.target.value))}
                                        style={{ ...styles.select, width: '45px', textAlign: 'center' }}
                                    />
                                </div>
                            </div>
                        </div>

                        {loading ? (
                            <div style={styles.loadingContainer}>
                                <Loader2 size={32} color="#16a34a" className="spinner" />
                                <span>Génération du graphique de décomposition...</span>
                            </div>
                        ) : decompResult ? (
                            <div style={{ width: '100%', height: '650px' }}>
                                <Plot
                                    data={[
                                        { x: decompResult.timestamps, y: decompResult.observed, type: 'scatter', mode: 'lines', name: 'Observé', line: { color: '#2563eb', width: 1.8 }, xaxis: 'x', yaxis: 'y' },
                                        { x: decompResult.timestamps, y: decompResult.trend, type: 'scatter', mode: 'lines', name: 'Tendance', line: { color: '#dc2626', width: 1.8 }, xaxis: 'x2', yaxis: 'y2' },
                                        { x: decompResult.timestamps, y: decompResult.seasonal, type: 'scatter', mode: 'lines', name: 'Saisonnalité', line: { color: '#16a34a', width: 1.8 }, xaxis: 'x3', yaxis: 'y3' },
                                        { x: decompResult.timestamps, y: decompResult.residual, type: 'scatter', mode: 'lines', name: 'Résidus', line: { color: '#9333ea', width: 1.8 }, xaxis: 'x4', yaxis: 'y4' },
                                    ]}
                                    layout={{
                                        grid: { rows: 4, columns: 1, pattern: 'independent' },
                                        autosize: true,
                                        showlegend: false,
                                        margin: { t: 20, b: 30, l: 60, r: 20 },
                                        yaxis: { title: { text: 'Observé', font: { size: 11, color: '#475569', weight: 'bold' } }, tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' },
                                        yaxis2: { title: { text: 'Tendance', font: { size: 11, color: '#475569', weight: 'bold' } }, tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' },
                                        yaxis3: { title: { text: 'Saisonnalité', font: { size: 11, color: '#475569', weight: 'bold' } }, tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' },
                                        yaxis4: { title: { text: 'Résidus', font: { size: 11, color: '#475569', weight: 'bold' } }, tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' },
                                        xaxis4: { tickfont: { size: 10, color: '#64748b' } }
                                    }}
                                    useResizeHandler={true}
                                    style={{ width: '100%', height: '100%' }}
                                    config={{ displayModeBar: false }}
                                />
                            </div>
                        ) : (
                            <div style={{ display: 'flex', justifyContent: 'center', padding: '40px', color: '#94a3b8', fontSize: '13px', fontStyle: 'italic' }}>
                                Aucune donnée de décomposition disponible.
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'acf-pacf' && (
                    <div style={styles.card}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <div style={styles.iconBg('#faf5ff', '#9333ea')}>
                                    <BarChart2 size={20} />
                                </div>
                                <div>
                                    <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a' }}>Fonctions d'Autocorrélation (ACF / PACF)</h2>
                                    <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>Identification des retardations (Lags) statistiquement significatives</p>
                                </div>
                            </div>

                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                                <div style={{ display: 'flex', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '4px', borderRadius: '16px', gap: '4px' }}>
                                    <button onClick={() => setPlotType('acf')} style={styles.radioBtn(plotType === 'acf')}>ACF</button>
                                    <button onClick={() => setPlotType('pacf')} style={styles.radioBtn(plotType === 'pacf')}>PACF</button>
                                </div>

                                <div style={{ ...styles.selectBox, minWidth: '120px', maxWidth: '140px' }}>
                                    <span style={{ color: '#64748b', fontWeight: '600', flexShrink: 0 }}>Lags :</span>
                                    <input
                                        type="number"
                                        value={lags}
                                        onChange={(e) => setLags(Number(e.target.value))}
                                        style={{ ...styles.select, width: '45px', textAlign: 'center' }}
                                    />
                                </div>
                            </div>
                        </div>

                        {loading ? (
                            <div style={styles.loadingContainer}>
                                <Loader2 size={32} color="#9333ea" className="spinner" />
                                <span>Calcul du graphique ACF / PACF...</span>
                            </div>
                        ) : acfResult ? (
                            <div style={{ width: '100%', height: '420px' }}>
                                <Plot
                                    data={[
                                        { x: acfResult.lags, y: acfResult.confidence_interval_upper, type: 'scatter', mode: 'lines', line: { color: 'transparent' }, showlegend: false },
                                        { x: acfResult.lags, y: acfResult.confidence_interval_lower, type: 'scatter', mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(59, 130, 246, 0.12)', line: { color: 'transparent' }, name: 'IC 95%' },
                                        { x: acfResult.lags, y: acfResult.values, type: 'bar', marker: { color: '#2563eb' }, name: acfResult.plot_type.toUpperCase() }
                                    ]}
                                    layout={{
                                        autosize: true,
                                        margin: { t: 20, b: 50, l: 60, r: 20 },
                                        xaxis: { title: { text: 'Retardations (Lags)', font: { size: 11, color: '#475569', weight: 'bold' } }, tickfont: { size: 10, color: '#64748b' } },
                                        yaxis: { title: { text: 'Autocorrélation', font: { size: 11, color: '#475569', weight: 'bold' } }, range: [-1.1, 1.1], tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' },
                                        showlegend: false
                                    }}
                                    useResizeHandler={true}
                                    style={{ width: '100%', height: '100%' }}
                                    config={{ displayModeBar: false }}
                                />
                            </div>
                        ) : (
                            <div style={{ display: 'flex', justifyContent: 'center', padding: '40px', color: '#94a3b8', fontSize: '13px', fontStyle: 'italic' }}>
                                Aucune donnée ACF / PACF disponible.
                            </div>
                        )}
                    </div>
                )}

            </div>
        </div>
    );
}