import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
    Sliders, Filter, AlertTriangle, ChevronDown,
    Layers, AlertCircle, Hash, TrendingDown, Activity, Columns
} from 'lucide-react';
import Plot from 'react-plotly.js';

// Hàm tự động tạo màu phân bố đều theo không gian HSL cho các version (tương tự SingleColumn)
const getDynamicColor = (index, total) => {
    if (total <= 1) return '#059669';
    const hue = Math.round((index * 360) / total);
    return `hsl(${hue}, 70%, 45%)`;
};

export default function MultiColumnProfiling({ initialTargetVar = 'visitor_count' }) {
    // --- States Quản lý Bộ lọc & Version ---
    const [selectedVersion, setSelectedVersion] = useState('v0_raw');
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [targetVar, setTargetVar] = useState(initialTargetVar);
    const [corrMethod, setCorrMethod] = useState('pearson');
    const [varX, setVarX] = useState('datetime');
    const [varY, setVarY] = useState('visitor_count');

    // --- States Dữ liệu & List Metadata ---
    const [data, setData] = useState(null);
    const [allVersions, setAllVersions] = useState(['v0_raw']);
    const [attractionList, setAttractionList] = useState([]);
    const [availableColumns, setAvailableColumns] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const abortControllerRef = useRef(null);

    // --- 1. Lấy danh sách Version và Attraction khả dụng từ Server ---
    useEffect(() => {
        const fetchMeta = async () => {
            try {
                // Fetch danh sách Versions
                const resSchema = await fetch('http://localhost:8000/analysis/versions-schema');
                if (resSchema.ok) {
                    const resData = await resSchema.json();
                    const list = resData.versions || resData || [];
                    const vIds = list.map(v => typeof v === 'string' ? v : (v.version_id || v.table_name));
                    if (!vIds.includes('v0_raw')) vIds.unshift('v0_raw');
                    setAllVersions([...new Set(vIds)]);
                }

                // Fetch danh sách Attractions
                const resAttr = await fetch('http://localhost:8000/analysis/attractions');
                if (resAttr.ok) {
                    const listAttr = await resAttr.json();
                    setAttractionList(listAttr);
                }
            } catch (err) {
                console.warn("Không thể tải metadata bộ lọc:", err);
            }
        };

        fetchMeta();
    }, []);

    // --- 2. Lọc danh sách Version tương thích theo Attraction ---
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

    // --- 3. Xử lý khi thay đổi Attraction ---
    const handleAttractionChange = (newAttr) => {
        setSelectedAttraction(newAttr);

        let availableForNewAttr = allVersions;
        if (newAttr && newAttr !== 'ALL') {
            const attrUpper = newAttr.toUpperCase();
            availableForNewAttr = allVersions.filter(v => v === 'v0_raw' || v.toUpperCase().includes(attrUpper));
        }

        if (!availableForNewAttr.includes(selectedVersion)) {
            setSelectedVersion(availableForNewAttr[0] || 'v0_raw');
        }
    };

    // --- 4. Fetch Dữ liệu Multi-Column Profiling ---
    const fetchMultiColumnData = async (ver, attr, target, method, signal) => {
        setLoading(true);
        setError(null);
        try {
            let url = `http://localhost:8000/analysis/multi-column-profiling?version=${ver}&target_var=${target}&corr_method=${method.toLowerCase()}`;
            if (attr && attr !== 'ALL') {
                url += `&id_attraction=${attr}`;
            }

            const response = await fetch(url, { signal });
            if (!response.ok) {
                const errJson = await response.json().catch(() => ({}));
                throw new Error(errJson.detail || `Lỗi HTTP: ${response.status}`);
            }
            const resData = await response.json();
            setData(resData);

            // Cập nhật các cột khả dụng cho Dropdown Cột mục tiêu
            if (resData.correlation_matrix?.columns?.length > 0) {
                const colsList = resData.correlation_matrix.columns;
                setAvailableColumns(colsList);

                if (!colsList.includes(target)) {
                    setTargetVar(colsList[0]);
                }
                setVarY(colsList[0]);
            }
        } catch (err) {
            if (err.name !== 'AbortError') {
                setError(err.message || "Không thể tải dữ liệu phân tích đa cột.");
            }
        } finally {
            if (!signal.aborted) setLoading(false);
        }
    };

    useEffect(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        const controller = new AbortController();
        abortControllerRef.current = controller;

        fetchMultiColumnData(selectedVersion, selectedAttraction, targetVar, corrMethod, controller.signal);

        return () => controller.abort();
    }, [selectedVersion, selectedAttraction, targetVar, corrMethod]);

    // --- Derived Data ---
    const cols = useMemo(() => data?.correlation_matrix?.columns || [], [data]);
    const matrixValues = useMemo(() => data?.correlation_matrix?.values || [], [data]);
    const targetCorr = useMemo(() => data?.target_correlation || [], [data]);
    const funcDeps = useMemo(() => data?.functional_dependencies || [], [data]);
    const sampleData = useMemo(() => data?.sample_data || [], [data]);

    // --- Styles đồng bộ theo SingleColumn & DataAnalysis ---
    const styles = {
        wrapper: { minHeight: '100vh', backgroundColor: '#f8fafc', padding: '24px', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1e293b' },
        container: { maxWidth: '1360px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' },
        card: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)', boxSizing: 'border-box' },
        headerCard: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '16px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' },
        topBar: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' },
        selectBox: { display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '8px 14px', borderRadius: '16px', fontSize: '12px', minWidth: '180px' },
        select: { border: 'none', background: 'transparent', fontWeight: 'bold', color: '#0f172a', outline: 'none', cursor: 'pointer', appearance: 'none', width: '100%', lineHeight: '1' },
        kpiGrid: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '12px 16px', display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' },
        kpiItem: { display: 'flex', alignItems: 'center', gap: '12px', padding: '8px 4px', minWidth: 0 },
        iconBg: (bg, color) => ({ backgroundColor: bg, color: color, padding: '10px', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }),
        gridTwoCols: { display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '16px' },
        tableTh: { padding: '10px 0', borderBottom: '1px solid #f1f5f9', color: '#94a3b8', fontSize: '11px', fontWeight: '600', textAlign: 'left' },
        tableTd: { padding: '10px 0', borderBottom: '1px solid #f8fafc', fontSize: '12px' },
        loadingOverlay: { display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', width: '100%', backgroundColor: 'rgba(255, 255, 255, 0.7)', position: 'absolute', top: 0, left: 0, borderRadius: '24px', zIndex: 10 }
    };

    if (loading && !data) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '500px', gap: '16px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '50%', border: '4px solid #dbeafe', borderTopColor: '#2563eb', animation: 'spin 1s linear infinite' }} />
                <p style={{ color: '#64748b', fontWeight: '500', fontSize: '14px' }}>Chargement du profilage multi-colonnes...</p>
                <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
            </div>
        );
    }

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* 1. HEADER CARD & STEPPER VERSION */}
                <div style={styles.headerCard}>
                    <div style={styles.topBar}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={styles.iconBg('#eff6ff', '#2563eb')}>
                                <Columns size={24} />
                            </div>
                            <div>
                                <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2' }}>
                                    Profilage Multi-Colonnes
                                </h1>
                                <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', lineHeight: '1.2' }}>
                                    Target : <span style={{ color: '#2563eb' }}>{targetVar}</span> | Version : <span style={{ color: '#059669' }}>{selectedVersion}</span>
                                </p>
                            </div>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                            {/* Filtre Attraction */}
                            <div style={styles.selectBox}>
                                <Filter size={14} color="#94a3b8" />
                                <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap' }}>Attraction :</span>
                                <select value={selectedAttraction} onChange={(e) => handleAttractionChange(e.target.value)} style={styles.select}>
                                    <option value="ALL">Toutes les attractions</option>
                                    {attractionList.map(id => <option key={id} value={id}>Attraction {id}</option>)}
                                </select>
                                <ChevronDown size={14} color="#94a3b8" />
                            </div>

                            {/* Filtre Variable Cột mục tiêu */}
                            <div style={styles.selectBox}>
                                <Sliders size={14} color="#94a3b8" />
                                <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap' }}>Variable :</span>
                                <select value={targetVar} onChange={(e) => setTargetVar(e.target.value)} style={styles.select}>
                                    {availableColumns.length > 0 ? (
                                        availableColumns.map(col => <option key={col} value={col}>{col}</option>)
                                    ) : (
                                        <option value={targetVar}>{targetVar}</option>
                                    )}
                                </select>
                                <ChevronDown size={14} color="#94a3b8" />
                            </div>
                        </div>
                    </div>

                    {/* STEPPER CHỌN VERSION */}
                    <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', border: '1px solid #e2e8f0', padding: '20px 24px', width: '100%', boxSizing: 'border-box' }}>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#1e293b', marginBottom: '20px' }}>
                            Choisissez la version principale des données
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

                {/* Báo lỗi hệ thống */}
                {error && (
                    <div style={{ padding: '16px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '16px', color: '#991b1b', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <AlertTriangle size={20} color="#dc2626" />
                        <span>{error}</span>
                    </div>
                )}

                {/* 2. KPI METRICS GRID */}
                <div style={styles.kpiGrid}>
                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#faf5ff', '#9333ea')}><Layers size={18} /></div>
                        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Colonnes</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#0f172a', marginTop: '2px', lineHeight: '1.2' }}>{loading ? '-' : cols.length}</div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fffbeb', '#d97706')}><AlertCircle size={18} /></div>
                        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Corrélations</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#d97706', marginTop: '2px', lineHeight: '1.2' }}>{loading ? '-' : targetCorr.length}</div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#ecfdf5', '#059669')}><Hash size={18} /></div>
                        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Dépendances</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#059669', marginTop: '2px', lineHeight: '1.2' }}>{loading ? '-' : funcDeps.length}</div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fff1f2', '#e11d48')}><TrendingDown size={18} /></div>
                        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Méthode</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#e11d48', textTransform: 'capitalize', marginTop: '2px', lineHeight: '1.2' }}>{corrMethod}</div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fef2f2', '#dc2626')}><Activity size={18} /></div>
                        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Échantillons</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#dc2626', marginTop: '2px', lineHeight: '1.2' }}>{loading ? '-' : sampleData.length}</div>
                        </div>
                    </div>
                </div>

                {/* 3. HEATMAP CORRELATION MATRIX (ĐÃ TỐI ƯU RỘNG RÃI & KHUNG METHODE ĐẸP) */}
                <div style={{ ...styles.card, position: 'relative', minHeight: '680px' }}>
                    {loading && (
                        <div style={styles.loadingOverlay}>
                            <span style={{ fontSize: '13px', color: '#64748b', fontWeight: '600' }}>Chargement de la matrice...</span>
                        </div>
                    )}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
                        <div>
                            <h2 style={{ margin: 0, fontSize: '16px', fontWeight: 'bold', color: '#0f172a', lineHeight: '1.2' }}>
                                Matrice de Corrélation
                            </h2>
                            <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>
                                Visualisation des corrélations croisées entre les variables
                            </p>
                        </div>

                        {/* KHUNG METHODE ĐƯỢC CHUẨN HÓA ĐẸP MẮT */}
                        <div style={styles.selectBox}>
                            <Sliders size={14} color="#94a3b8" />
                            <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap' }}>Méthode :</span>
                            <select value={corrMethod} onChange={(e) => setCorrMethod(e.target.value)} style={styles.select}>
                                {['pearson', 'spearman', 'kendall'].map(m => (
                                    <option key={m} value={m}>{m.charAt(0).toUpperCase() + m.slice(1)}</option>
                                ))}
                            </select>
                            <ChevronDown size={14} color="#94a3b8" />
                        </div>
                    </div>

                    {/* VÙNG CHỨA HEATMAP MỞ RỘNG TOÀN DIỆN */}
                    <div style={{ width: '100%', height: '580px' }}>
                        {cols.length > 0 ? (
                            <Plot
                                data={[{
                                    z: matrixValues, x: cols, y: cols, type: 'heatmap',
                                    colorscale: [[0.0, '#2563eb'], [0.5, '#f8fafc'], [1.0, '#1d4ed8']],
                                    zmin: -1, zmax: 1,
                                    colorbar: {
                                        thickness: 14,
                                        len: 0.95,
                                        tickfont: { size: 11, color: '#64748b' },
                                        outlinewidth: 0
                                    },
                                    hovertemplate: '<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Corrélation:</b> %{z:.3f}<extra></extra>'
                                }]}
                                layout={{
                                    autosize: true,
                                    margin: { t: 30, r: 30, l: 110, b: 110 },
                                    xaxis: { tickangle: -35, tickfont: { size: 11, color: '#475569' }, automargin: true },
                                    yaxis: { tickfont: { size: 11, color: '#475569' }, autorange: 'reversed', automargin: true }
                                }}
                                useResizeHandler={true}
                                style={{ width: '100%', height: '100%' }}
                                config={{ displayModeBar: false }}
                            />
                        ) : (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8', fontStyle: 'italic', fontSize: '13px' }}>
                                Aucune donnée de corrélation disponible.
                            </div>
                        )}
                    </div>
                </div>

                {/* 4. TABLES DÂN CORRELATION */}
                <div style={styles.gridTwoCols}>
                    <div style={{ ...styles.card, minHeight: '380px' }}>
                        <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: 'bold', color: '#0f172a' }}>
                            Fortes Corrélations avec <span style={{ color: '#2563eb' }}>{targetVar}</span>
                        </h3>
                        <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                <thead>
                                    <tr>
                                        <th style={styles.tableTh}>Colonne</th>
                                        <th style={{ ...styles.tableTh, textAlign: 'right' }}>Corrélation</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {targetCorr.map((item, idx) => (
                                        <tr key={idx}>
                                            <td style={{ ...styles.tableTd, color: '#334155', fontWeight: '500' }}>{item.colonne}</td>
                                            <td style={{ ...styles.tableTd, textAlign: 'right', fontFamily: 'monospace', fontWeight: 'bold', color: '#0f172a' }}>
                                                {item.correlation !== null ? item.correlation.toFixed(6) : 'N/A'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div style={{ ...styles.card, minHeight: '380px' }}>
                        <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: 'bold', color: '#0f172a' }}>
                            Corrélations Inverses / Faibles
                        </h3>
                        <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                <thead>
                                    <tr>
                                        <th style={styles.tableTh}>Colonne</th>
                                        <th style={{ ...styles.tableTh, textAlign: 'right' }}>Corrélation</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {targetCorr.slice().reverse().map((item, idx) => (
                                        <tr key={idx}>
                                            <td style={{ ...styles.tableTd, color: '#334155', fontWeight: '500' }}>{item.colonne}</td>
                                            <td style={{ ...styles.tableTd, textAlign: 'right', fontFamily: 'monospace', fontWeight: 'bold', color: '#0f172a' }}>
                                                {item.correlation !== null ? item.correlation.toFixed(6) : 'N/A'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* 5. INTERACTIONS LINE PLOT */}
                <div style={{ ...styles.card, minHeight: '400px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
                        <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a', lineHeight: '1.2' }}>
                            Interactions entre Variables
                        </h2>
                        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                            <div style={styles.selectBox}>
                                <span style={{ color: '#94a3b8', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>
                                    Axe X :
                                </span>
                                <select value={varX} onChange={(e) => setVarX(e.target.value)} style={styles.select}>
                                    <option value="datetime">datetime</option>
                                    {cols.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                                <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                            </div>

                            <div style={styles.selectBox}>
                                <span style={{ color: '#94a3b8', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>
                                    Axe Y :
                                </span>
                                <select value={varY} onChange={(e) => setVarY(e.target.value)} style={styles.select}>
                                    {cols.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                                <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                            </div>
                        </div>
                    </div>

                    <div style={{ width: '100%', height: '320px' }}>
                        {sampleData.length > 0 ? (
                            <Plot
                                data={[{
                                    x: sampleData.map(d => d[varX] || d.datetime || d.date || ''),
                                    y: sampleData.map(d => d[varY]),
                                    type: 'scatter', mode: 'lines',
                                    line: { color: '#2563eb', width: 2 },
                                    name: varY
                                }]}
                                layout={{
                                    autosize: true, margin: { t: 20, r: 20, l: 50, b: 50 },
                                    xaxis: { tickfont: { size: 10, color: '#64748b' } },
                                    yaxis: { tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' },
                                    showlegend: true, legend: { x: 0.88, y: 1 }
                                }}
                                useResizeHandler={true}
                                style={{ width: '100%', height: '100%' }}
                                config={{ displayModeBar: false }}
                            />
                        ) : (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8', fontStyle: 'italic', fontSize: '12px' }}>
                                Aucune donnée temporelle disponible.
                            </div>
                        )}
                    </div>
                </div>

                {/* 6. FUNCTIONAL DEPENDENCIES CARD */}
                <div style={{ ...styles.card, padding: '32px', minHeight: '220px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '20px' }}>
                        <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 'bold', color: '#0f172a' }}>Functional Dependencies</h2>
                        <div style={{ position: 'relative', maxWidth: '480px' }}>
                            <label style={{ position: 'absolute', top: '-8px', left: '16px', backgroundColor: '#ffffff', padding: '0 4px', fontSize: '11px', fontWeight: '600', color: '#94a3b8' }}>
                                Target Variable
                            </label>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', border: '1px solid #e2e8f0', borderRadius: '16px', fontSize: '14px', fontWeight: '600', color: '#1e293b' }}>
                                <span>{targetVar}</span>
                                <ChevronDown size={16} color="#94a3b8" />
                            </div>
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px', padding: '16px 0' }}>
                        {funcDeps.length > 0 ? (
                            funcDeps.map((dep, idx) => (
                                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', fontWeight: '500' }}>
                                    <span style={{ fontFamily: 'monospace', color: '#475569' }}>
                                        &#123;{dep.determinants ? dep.determinants.join(', ') : dep.determinant}&#125;
                                    </span>
                                    <span style={{ color: '#ef4444', fontWeight: 'bold' }}>→</span>
                                    <span style={{ color: '#2563eb', fontWeight: '600' }}>{dep.dependent}</span>
                                </div>
                            ))
                        ) : (
                            <div style={{ textAlign: 'center', color: '#94a3b8', fontStyle: 'italic', fontSize: '12px', padding: '16px 0' }}>
                                No functional dependencies found for {targetVar}.
                            </div>
                        )}
                    </div>
                </div>

            </div>
        </div>
    );
}