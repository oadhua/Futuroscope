import React, { useState, useEffect, useMemo } from 'react';
import {
    Sliders, Filter, AlertTriangle, ChevronDown,
    Layers, AlertCircle, Hash, TrendingDown, Activity, Columns
} from 'lucide-react';
import Plot from 'react-plotly.js';

export default function MultiColumnProfiling({ initialTargetVar = 'visitor_count' }) {
    // --- States ---
    const [targetVar, setTargetVar] = useState(initialTargetVar);
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [corrMethod, setCorrMethod] = useState('pearson');
    const [varX, setVarX] = useState('datetime');
    const [varY, setVarY] = useState('visitor_count');

    const [data, setData] = useState(null);
    const [availableAttractions, setAvailableAttractions] = useState([]);
    const [availableColumns, setAvailableColumns] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // --- Fetch Metadata ---
    useEffect(() => {
        const fetchMeta = async () => {
            try {
                let url = 'http://localhost:8000/analysis/analyse-globale?nom_table=fact_attraction_hourly&nb_lignes_apercu=1';
                if (selectedAttraction && selectedAttraction !== 'ALL') {
                    url += `&id_attraction=${selectedAttraction}`;
                }

                const response = await fetch(url);
                if (response.ok) {
                    const resGlobal = await response.json();

                    if (resGlobal.comparaison_colonnes) {
                        const cols = resGlobal.comparaison_colonnes.map(c => c.nom_colonne);
                        setAvailableColumns(cols);
                        setTargetVar(prev => (cols.length > 0 && !cols.includes(prev) ? cols[0] : prev));
                    }

                    if (resGlobal.liste_attractions && availableAttractions.length === 0) {
                        setAvailableAttractions(resGlobal.liste_attractions);
                    }
                }
            } catch (err) {
                console.warn("Méta-données non disponibles:", err);
            }
        };

        fetchMeta();
    }, [selectedAttraction]);

    // --- Fetch Multi-Column Data ---
    const fetchMultiColumnData = async () => {
        setLoading(true);
        setError(null);
        try {
            let url = `http://localhost:8000/analysis/multi-column-profiling?target_var=${targetVar}&corr_method=${corrMethod.toLowerCase()}`;
            if (selectedAttraction && selectedAttraction !== 'ALL') {
                url += `&id_attraction=${selectedAttraction}`;
            }

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            const resData = await response.json();
            setData(resData);

            if (resData.correlation_matrix?.columns?.length > 0) {
                setVarY(resData.correlation_matrix.columns[0]);
            }
        } catch (err) {
            setError(err.message || "Impossible de charger le profilage multi-colonnes.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchMultiColumnData();
    }, [targetVar, selectedAttraction, corrMethod]);

    // --- Derived Data ---
    const cols = useMemo(() => data?.correlation_matrix?.columns || [], [data]);
    const matrixValues = useMemo(() => data?.correlation_matrix?.values || [], [data]);
    const targetCorr = useMemo(() => data?.target_correlation || [], [data]);
    const funcDeps = useMemo(() => data?.functional_dependencies || [], [data]);
    const sampleData = useMemo(() => data?.sample_data || [], [data]);

    // --- Styles được tối ưu hóa chuẩn Layout ---
    const styles = {
        wrapper: { minHeight: '100vh', backgroundColor: '#f8fafc', padding: '24px', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1e293b' },
        container: { maxWidth: '1150px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '16px' },
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
            minHeight: '80px',            // Đã sửa thuộc tính minHeight chính xác
            boxSizing: 'border-box',
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
            minWidth: '220px',
            maxWidth: '280px',             // Giới hạn chiều rộng thẻ select vừa đủ đẹp
            flexShrink: 1
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
            textOverflow: 'ellipsis',      // Cắt bớt văn bản bằng dấu ... nếu tên quá dài
            overflow: 'hidden',
            whiteSpace: 'nowrap'
        },
        kpiGrid: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '12px', display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)', minHeight: '80px', boxSizing: 'border-box' },
        kpiItem: { display: 'flex', alignItems: 'center', gap: '12px', padding: '8px 10px', minWidth: 0 },
        iconBg: (bg, color) => ({ backgroundColor: bg, color: color, padding: '10px', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }),
        gridTwoCols: { display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '16px' },
        tableTh: { padding: '10px 0', borderBottom: '1px solid #f1f5f9', color: '#94a3b8', fontSize: '11px', fontWeight: '600', textAlign: 'left' },
        tableTd: { padding: '10px 0', borderBottom: '1px solid #f8fafc', fontSize: '12px' },
        loadingOverlay: { display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', width: '100%', backgroundColor: 'rgba(255, 255, 255, 0.7)', position: 'absolute', top: 0, left: 0, borderRadius: '24px', zIndex: 10 }
    };

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* 1. HEADER (Tuyệt đối không đè chữ, không rớt dòng) */}
                <div style={styles.headerCard}>
                    {/* Cụm Tiêu đề bên trái: Co giãn thông minh */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flex: '1 1 auto', minWidth: 0 }}>
                        <div style={styles.iconBg('#eff6ff', '#2563eb')}>
                            <Columns size={24} />
                        </div>
                        <div style={{ minWidth: 0, overflow: 'hidden' }}>
                            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.02em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                Profilage Multi-Colonnes
                            </h1>
                            <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                Variable sélectionnée : <span style={{ color: '#2563eb' }}>{targetVar}</span>
                            </p>
                        </div>
                    </div>

                    {/* Cụm Select Filters bên phải: Cố định kích thước */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0 }}>
                        {/* Dropdown 1: Colonne */}
                        <div style={styles.selectBox}>
                            <Sliders size={15} color="#94a3b8" style={{ flexShrink: 0 }} />
                            <span style={{ color: '#64748b', fontWeight: '600', flexShrink: 0 }}>Colonne :</span>
                            <select value={targetVar} onChange={(e) => setTargetVar(e.target.value)} style={styles.select}>
                                {availableColumns.length > 0 ? (
                                    availableColumns.map(col => <option key={col} value={col}>{col}</option>)
                                ) : (
                                    <option value={targetVar}>{targetVar}</option>
                                )}
                            </select>
                            <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                        </div>

                        {/* Dropdown 2: Attraction */}
                        <div style={styles.selectBox}>
                            <Filter size={15} color="#94a3b8" style={{ flexShrink: 0 }} />
                            <span style={{ color: '#64748b', fontWeight: '600', flexShrink: 0 }}>Attraction :</span>
                            <select value={selectedAttraction} onChange={(e) => setSelectedAttraction(e.target.value)} style={styles.select}>
                                <option value="ALL">Toutes les attractions</option>
                                {availableAttractions.map(id => <option key={id} value={id}>Attraction {id}</option>)}
                            </select>
                            <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                        </div>
                    </div>
                </div>

                {/* 2. KPI STATS CARD */}
                <div style={styles.kpiGrid}>
                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#faf5ff', '#9333ea')}><Layers size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Colonnes</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#0f172a', marginTop: '2px' }}>{loading ? '-' : cols.length}</div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fffbeb', '#d97706')}><AlertCircle size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Corrélations</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#0f172a', marginTop: '2px' }}>{loading ? '-' : targetCorr.length}</div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#ecfdf5', '#059669')}><Hash size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Dépendances</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#059669', marginTop: '2px' }}>{loading ? '-' : funcDeps.length}</div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fff1f2', '#e11d48')}><TrendingDown size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Méthode</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#e11d48', textTransform: 'capitalize', marginTop: '2px' }}>{corrMethod}</div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fef2f2', '#dc2626')}><Activity size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Échantillons</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#dc2626', marginTop: '2px' }}>{loading ? '-' : sampleData.length}</div>
                        </div>
                    </div>
                </div>

                {/* Báo lỗi nếu API gặp sự cố */}
                {error && (
                    <div style={{ padding: '16px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '16px', color: '#991b1b', fontSize: '13px' }}>
                        Erreur de chargement: {error}
                    </div>
                )}

                {/* 3. HEATMAP CORRELATION MATRIX */}
                <div style={{ ...styles.card, position: 'relative', minHeight: '520px' }}>
                    {loading && (
                        <div style={styles.loadingOverlay}>
                            <span style={{ fontSize: '13px', color: '#64748b', fontWeight: '600' }}>Chargement de la matrice...</span>
                        </div>
                    )}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
                        <h2 style={{ margin: 0, fontSize: '16px', fontWeight: 'bold', color: '#0f172a' }}>Matrice de Corrélation</h2>
                        <div style={styles.selectBox}>
                            <span style={{ color: '#94a3b8' }}>Méthode:</span>
                            <select value={corrMethod} onChange={(e) => setCorrMethod(e.target.value)} style={styles.select}>
                                {['pearson', 'spearman', 'kendall'].map(m => <option key={m} value={m}>{m}</option>)}
                            </select>
                        </div>
                    </div>

                    <div style={{ width: '100%', height: '420px' }}>
                        {cols.length > 0 ? (
                            <Plot
                                data={[{
                                    z: matrixValues, x: cols, y: cols, type: 'heatmap',
                                    colorscale: [[0.0, '#2563eb'], [0.5, '#f8fafc'], [1.0, '#1d4ed8']],
                                    zmin: -1, zmax: 1,
                                    colorbar: { thickness: 12, len: 0.88, tickfont: { size: 10, color: '#64748b' } },
                                    hovertemplate: '<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Corrélation:</b> %{z:.3f}<extra></extra>'
                                }]}
                                layout={{
                                    autosize: true, margin: { t: 20, r: 20, l: 90, b: 90 },
                                    xaxis: { tickangle: -40, tickfont: { size: 10, color: '#64748b' }, automargin: true },
                                    yaxis: { tickfont: { size: 10, color: '#64748b' }, autorange: 'reversed', automargin: true }
                                }}
                                useResizeHandler={true}
                                style={{ width: '100%', height: '100%' }}
                                config={{ displayModeBar: false }}
                            />
                        ) : (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8', fontStyle: 'italic', fontSize: '12px' }}>
                                Aucune donnée de corrélation disponible.
                            </div>
                        )}
                    </div>
                </div>

                {/* 4. SIDE-BY-SIDE TABLES */}
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
                        <h2 style={{ margin: 0, fontSize: '16px', fontWeight: 'bold', color: '#0f172a' }}>Interactions entre Variables</h2>
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
                        <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 'bold', color: '#0f172a' }}>Functional Dependancies</h2>
                        <div style={{ position: 'relative', maxWidth: '480px' }}>
                            <label style={{ position: 'absolute', top: '-8px', left: '16px', backgroundColor: '#ffffff', padding: '0 4px', fontSize: '11px', fontWeight: '600', color: '#94a3b8' }}>
                                Choose The Dependante Column
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