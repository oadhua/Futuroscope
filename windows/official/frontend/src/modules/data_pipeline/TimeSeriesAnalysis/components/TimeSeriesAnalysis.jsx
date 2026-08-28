import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
    Activity, TrendingUp, Sliders, Filter,
    ChevronDown, Play, CheckCircle2, AlertCircle,
    BarChart2, Layers, Clock, Loader2, AlertTriangle
} from 'lucide-react';
import Plot from 'react-plotly.js';

const API_BASE_URL = 'http://localhost:8000/time-series';

// Hàm tự động tạo màu phân bố đều theo không gian HSL cho các version (đồng bộ từ SingleColumn)
const getDynamicColor = (index, total) => {
    if (total <= 1) return '#2563eb';
    const hue = Math.round((index * 360) / total);
    return `hsl(${hue}, 70%, 45%)`;
};

export default function TimeSeriesAnalysis() {
    const [activeTab, setActiveTab] = useState('stationarity');

    // Filter States
    const [selectedVersion, setSelectedVersion] = useState('v0_raw');
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [selectedColumn, setSelectedColumn] = useState('');

    // Dynamic Options States
    const [availableAttractions, setAvailableAttractions] = useState([]);
    const [availableColumns, setAvailableColumns] = useState([]);
    const [allVersions, setAllVersions] = useState(['v0_raw']);
    const [loadingOptions, setLoadingOptions] = useState(true);

    // Analysis Execution States
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Tab Specific States
    const [statMethod, setStatMethod] = useState('adf');
    const [statResult, setStatResult] = useState(null);

    const [decompModel, setDecompModel] = useState('additive');
    const [decompPeriod, setDecompPeriod] = useState(24);
    const [decompResult, setDecompResult] = useState(null);

    const [plotType, setPlotType] = useState('acf');
    const [lags, setLags] = useState(25);
    const [acfResult, setAcfResult] = useState(null);

    const abortControllerRef = useRef(null);

    // --- 1. Lấy danh sách Schema/Versions khả dụng từ Backend ---
    useEffect(() => {
        const fetchVersions = async () => {
            try {
                const resSchema = await fetch('http://localhost:8000/analysis/versions-schema');
                if (resSchema.ok) {
                    const resData = await resSchema.json();
                    const list = resData.versions || resData || [];
                    const vIds = list.map(v => typeof v === 'string' ? v : (v.version_id || v.table_name));
                    if (!vIds.includes('v0_raw')) vIds.unshift('v0_raw');
                    setAllVersions([...new Set(vIds)]);
                }
            } catch (err) {
                console.warn("Méta-données des versions non disponibles:", err);
            }
        };

        fetchVersions();
    }, []);

    // --- 2. Lấy danh sách Attractions & Columns theo Version và Attraction được chọn ---
    useEffect(() => {
        const fetchMetaOptions = async () => {
            setLoadingOptions(true);
            try {
                let url = `${API_BASE_URL}/meta/options?version=${selectedVersion}`;

                if (selectedAttraction && selectedAttraction !== 'ALL') {
                    url += `&id_attraction=${selectedAttraction}`;
                } else {
                    url += `&id_attraction=ALL`;
                }

                const res = await fetch(url);
                if (res.ok) {
                    const data = await res.json();

                    // Cập nhật danh sách Attractions
                    if (data.attractions && data.attractions.length > 0) {
                        setAvailableAttractions(data.attractions);
                    }

                    // Cập nhật danh sách Cột (Features)
                    const cols = data.features || [];
                    setAvailableColumns(cols);

                    if (cols.length > 0) {
                        if (!selectedColumn || !cols.includes(selectedColumn)) {
                            setSelectedColumn(cols[0]);
                        }
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
    }, [selectedAttraction, selectedVersion]);

    // --- 3. Lọc danh sách Version phù hợp với Attraction được chọn ---
    const filteredVersionList = useMemo(() => {
        if (!selectedAttraction || selectedAttraction === 'ALL') {
            return allVersions;
        }
        const attrUpper = selectedAttraction.toUpperCase();
        const filtered = allVersions.filter(v => {
            if (v === 'v0_raw') return true;
            return v.toUpperCase().includes(attrUpper);
        });
        return filtered.length > 0 ? filtered : ['v0_raw'];
    }, [allVersions, selectedAttraction]);

    // Xử lý thay đổi Attraction
    const handleAttractionChange = (newAttr) => {
        setSelectedAttraction(newAttr);
        setSelectedColumn('');

        let availableForNewAttr = allVersions;
        if (newAttr && newAttr !== 'ALL') {
            const attrUpper = newAttr.toUpperCase();
            availableForNewAttr = allVersions.filter(v => v === 'v0_raw' || v.toUpperCase().includes(attrUpper));
        }

        if (!availableForNewAttr.includes(selectedVersion)) {
            setSelectedVersion(availableForNewAttr[0] || 'v0_raw');
        }
    };

    // --- 4. Fetch Data Thực thi tính toán ---
    const fetchData = async () => {
        if (!selectedColumn) return;

        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        const controller = new AbortController();
        abortControllerRef.current = controller;

        setLoading(true);
        setError(null);

        try {
            const params = new URLSearchParams({
                version: selectedVersion,
                col_name: selectedColumn,
            });

            if (selectedAttraction && selectedAttraction !== 'ALL') {
                params.append('id_attraction', selectedAttraction);
            } else {
                params.append('id_attraction', 'ALL');
            }

            if (activeTab === 'stationarity') {
                params.append('method', statMethod);
                const res = await fetch(`${API_BASE_URL}/stationarity?${params.toString()}`, { signal: controller.signal });
                if (!res.ok) throw new Error((await res.json()).detail || 'Erreur lors du calcul de la stationnarité');
                const data = await res.json();
                setStatResult(data);
            } else if (activeTab === 'decomposition') {
                params.append('model_type', decompModel);
                params.append('period', decompPeriod.toString());
                const res = await fetch(`${API_BASE_URL}/decomposition?${params.toString()}`, { signal: controller.signal });
                if (!res.ok) throw new Error((await res.json()).detail || 'Erreur lors de la décomposition');
                const data = await res.json();
                setDecompResult(data);
            } else if (activeTab === 'acf-pacf') {
                params.append('plot_type', plotType);
                params.append('lags', lags.toString());
                const res = await fetch(`${API_BASE_URL}/acf-pacf?${params.toString()}`, { signal: controller.signal });
                if (!res.ok) throw new Error((await res.json()).detail || 'Erreur lors du calcul ACF/PACF');
                const data = await res.json();
                setAcfResult(data);
            }
        } catch (err) {
            if (err.name !== 'AbortError') {
                setError(err.message || 'Impossible de charger les données analytiques');
            }
        } finally {
            if (!controller.signal.aborted) setLoading(false);
        }
    };

    // Tự động tính toán khi chuyển Tab hoặc đổi cấu hình chính
    useEffect(() => {
        if (!loadingOptions && selectedColumn) {
            fetchData();
        }
    }, [activeTab, selectedVersion, selectedColumn, statMethod, decompModel, decompPeriod, plotType, lags]);

    // --- Styles đồng bộ theo SingleColumn & DataAnalysis ---
    const styles = {
        wrapper: { minHeight: '100vh', backgroundColor: '#f8fafc', padding: '24px', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1e293b' },
        container: { maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' },
        card: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)', boxSizing: 'border-box' },
        headerCard: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '16px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' },
        topBar: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' },
        selectBox: { display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '8px 14px', borderRadius: '16px', fontSize: '12px', minWidth: '180px' },
        select: { border: 'none', background: 'transparent', fontWeight: 'bold', color: '#0f172a', outline: 'none', cursor: 'pointer', appearance: 'none', width: '100%', lineHeight: '1' },
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

                {/* 1. EN-TÊTE & FILTRES */}
                <div style={styles.headerCard}>
                    <div style={styles.topBar}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={styles.iconBg('#eff6ff', '#2563eb')}>
                                <Activity size={24} />
                            </div>
                            <div>
                                <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2' }}>
                                    Profilage et Analyse des Séries Temporelles
                                </h1>
                                <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', lineHeight: '1.2' }}>
                                    Variable : <span style={{ color: '#2563eb' }}>{selectedColumn || 'Chargement...'}</span> | Version : <span style={{ color: '#059669' }}>{selectedVersion}</span>
                                </p>
                            </div>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                            {/* Dropdown Attraction */}
                            <div style={styles.selectBox}>
                                <Filter size={15} color="#94a3b8" />
                                <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap' }}>Attraction :</span>
                                <select
                                    value={selectedAttraction}
                                    onChange={(e) => handleAttractionChange(e.target.value)}
                                    style={styles.select}
                                >
                                    <option value="ALL">Toutes les attractions</option>
                                    {availableAttractions.map(id => (
                                        <option key={id} value={id}>
                                            Attraction {id}
                                        </option>
                                    ))}
                                </select>
                                <ChevronDown size={14} color="#94a3b8" />
                            </div>

                            {/* Dropdown Colonne */}
                            <div style={styles.selectBox}>
                                <Sliders size={15} color="#94a3b8" />
                                <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap' }}>Colonne :</span>
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
                                <ChevronDown size={14} color="#94a3b8" />
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

                    {/* STEPPER CHỌN VERSION ĐỒNG BỘ THEO SINGLE COLUMN */}
                    <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', border: '1px solid #e2e8f0', padding: '20px 24px', width: '100%', boxSizing: 'border-box' }}>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#1e293b', marginBottom: '20px' }}>
                            Choisissez la version des données pour l'analyse temporelle
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between' }}>
                            {filteredVersionList.map((ver, index) => {
                                const isSelected = selectedVersion === ver;
                                const isLast = index === filteredVersionList.length - 1;
                                const verColor = getDynamicColor(index, filteredVersionList.length);

                                return (
                                    <React.Fragment key={ver}>
                                        <div
                                            onClick={() => setSelectedVersion(ver)}
                                            style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', userSelect: 'none', flexShrink: 0 }}
                                        >
                                            <div style={{
                                                width: '28px', height: '28px', borderRadius: '50%',
                                                backgroundColor: isSelected ? verColor : '#94a3b8',
                                                color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                fontSize: '12px', fontWeight: '700', transition: 'all 0.2s ease',
                                                boxShadow: isSelected ? `0 0 0 4px ${verColor}25` : 'none', flexShrink: 0
                                            }}>
                                                {index + 1}
                                            </div>
                                            <span style={{ fontSize: '13px', fontWeight: isSelected ? '700' : '500', color: isSelected ? '#1e293b' : '#64748b', whiteSpace: 'nowrap' }}>
                                                {ver}
                                            </span>
                                        </div>
                                        {!isLast && <div style={{ flex: 1, height: '1px', backgroundColor: '#e2e8f0', margin: '0 12px', minWidth: '16px' }} />}
                                    </React.Fragment>
                                );
                            })}
                        </div>
                    </div>
                </div>

                {error && (
                    <div style={{ padding: '16px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '16px', color: '#991b1b', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <AlertTriangle size={20} color="#dc2626" />
                        <span>{error}</span>
                    </div>
                )}

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