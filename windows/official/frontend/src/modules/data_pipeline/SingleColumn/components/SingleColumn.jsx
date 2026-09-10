import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
    BarChart3, AlertTriangle, Layers, Calendar,
    Filter, ChevronDown, Activity, TrendingDown,
    Clock, Sliders, Hash, BarChart2
} from 'lucide-react';
import Plot from 'react-plotly.js';

// Hàm tự động tạo màu phân bố đều theo không gian HSL cho vô số version
const getDynamicColor = (index, total) => {
    if (total <= 1) return '#059669';
    const hue = Math.round((index * 360) / total);
    return `hsl(${hue}, 70%, 45%)`;
};

export default function SingleColumn({ initialColumn = 'visitor_count' }) {
    // --- States ---
    const [selectedColumn, setSelectedColumn] = useState(initialColumn);
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [selectedVersion, setSelectedVersion] = useState('v0_raw');
    const [timeGranularity, setTimeGranularity] = useState('par_heure');

    const [data, setData] = useState(null);
    const [rawColumns, setRawColumns] = useState([]);
    const [attractionList, setAttractionList] = useState([]);
    const [allVersions, setAllVersions] = useState(['v0_raw']);

    // State lưu trữ dữ liệu chuỗi thời gian multi-version để vẽ so sánh
    const [multiVersionTimeSeries, setMultiVersionTimeSeries] = useState({});

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');

    const abortControllerRef = useRef(null);

    // --- 1. Filter Column List ---
    const availableColumns = useMemo(() => {
        if (!selectedAttraction || selectedAttraction === 'ALL') {
            return rawColumns;
        }

        const attrUpper = selectedAttraction.toUpperCase();
        const filtered = rawColumns.filter(col => {
            const colUpper = col.toUpperCase();
            const isMatchAttr = colUpper.includes(attrUpper);
            const isGeneralCol = !colUpper.match(/H\d{2}/i);
            return isMatchAttr || isGeneralCol;
        });

        return filtered.length > 0 ? filtered : rawColumns;
    }, [rawColumns, selectedAttraction]);

    // --- 2. Filter Versions List ---
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

    // --- 3. Handle Attraction Change ---
    const handleAttractionChange = (newAttr) => {
        setSelectedAttraction(newAttr);

        let newAvailableCols = rawColumns;
        if (newAttr && newAttr !== 'ALL') {
            const attrUpper = newAttr.toUpperCase();
            newAvailableCols = rawColumns.filter(col => {
                const colUpper = col.toUpperCase();
                return colUpper.includes(attrUpper) || !colUpper.match(/H\d{2}/i);
            });
        }

        if (newAvailableCols.length > 0 && !newAvailableCols.includes(selectedColumn)) {
            setSelectedColumn(newAvailableCols[0]);
        }

        let availableForNewAttr = allVersions;
        if (newAttr && newAttr !== 'ALL') {
            const attrUpper = newAttr.toUpperCase();
            availableForNewAttr = allVersions.filter(v => v === 'v0_raw' || v.toUpperCase().includes(attrUpper));
        }

        if (!availableForNewAttr.includes(selectedVersion)) {
            setSelectedVersion(availableForNewAttr[0] || 'v0_raw');
        }
    };

    // --- 4. Fetch Meta ---
    useEffect(() => {
        const fetchMeta = async () => {
            try {
                const resSchema = await fetch('http://localhost:8000/analysis/versions-schema');
                if (resSchema.ok) {
                    const resData = await resSchema.json();
                    const list = resData.versions || resData || [];
                    const vIds = list.map(v => typeof v === 'string' ? v : (v.version_id || v.table_name));
                    if (!vIds.includes('v0_raw')) vIds.unshift('v0_raw');
                    setAllVersions([...new Set(vIds)]);
                }

                let url = `http://localhost:8000/analysis/analyse-globale?nom_table=fact_attraction_hourly&nb_lignes_apercu=1&version=v0_raw`;
                const response = await fetch(url);
                if (response.ok) {
                    const resGlobal = await response.json();
                    if (resGlobal.comparaison_colonnes) {
                        const cols = resGlobal.comparaison_colonnes.map(c => c.nom_colonne);
                        setRawColumns(cols);
                    }
                    if (resGlobal.liste_attractions && resGlobal.liste_attractions.length > 0) {
                        setAttractionList(resGlobal.liste_attractions);
                    }
                }
            } catch (err) {
                console.warn("Méta-données non disponibles:", err);
            }
        };

        fetchMeta();
    }, []);

    // --- 5. Fetch Column Detail ---
    const fetchColumnDetail = async (col, ver, attr, signal) => {
        setLoading(true);
        setError(null);
        try {
            let url = `http://localhost:8000/analysis/colonne-detail?col_name=${col}&version=${ver}`;
            if (attr && attr !== 'ALL') {
                url += `&id_attraction=${attr}`;
            }

            const response = await fetch(url, { signal });
            if (!response.ok) {
                const errJson = await response.json().catch(() => ({}));
                throw new Error(errJson.detail || `Erreur HTTP: ${response.status}`);
            }
            const resData = await response.json();
            setData(resData);

            const targetVersions = filteredVersionList.length > 0 ? filteredVersionList : [ver];
            let multiUrl = `http://localhost:8000/analysis/multi-column-detail?col_name=${col}&versions=${targetVersions.join(',')}`;
            if (attr && attr !== 'ALL') {
                multiUrl += `&id_attraction=${attr}`;
            }

            const multiResponse = await fetch(multiUrl, { signal });
            if (multiResponse.ok) {
                const multiJson = await multiResponse.json();
                const versionsMapData = multiJson.versions_data || {};

                const tsMap = {};
                Object.keys(versionsMapData).forEach((vKey) => {
                    const vDetail = versionsMapData[vKey];
                    if (vDetail?.graphiques?.courbes_temporelles) {
                        tsMap[vKey] = vDetail.graphiques.courbes_temporelles;
                    }
                });
                setMultiVersionTimeSeries(tsMap);
            }

        } catch (err) {
            if (err.name !== 'AbortError') {
                setError(err.message || `Impossible de charger les détails pour la colonne ${col}`);
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

        if (selectedColumn) {
            fetchColumnDetail(selectedColumn, selectedVersion, selectedAttraction, controller.signal);
        }

        return () => controller.abort();
    }, [selectedColumn, selectedAttraction, selectedVersion, filteredVersionList]);

    // --- Derived Data ---
    const timeXAxisKey = useMemo(() => {
        switch (timeGranularity) {
            case 'par_heure': return 'heure';
            case 'par_jour': return 'date';
            case 'par_mois': return 'mois';
            case 'par_annee': return 'annee';
            default: return 'heure';
        }
    }, [timeGranularity]);

    const multiVersionTraces = useMemo(() => {
        const traces = [];
        const versionsToRender = filteredVersionList.length > 0 ? filteredVersionList : [selectedVersion];
        const totalVersions = versionsToRender.length;

        versionsToRender.forEach((ver, index) => {
            const verTimeSeries = multiVersionTimeSeries[ver]?.[timeGranularity];
            if (!Array.isArray(verTimeSeries) || verTimeSeries.length === 0) return;

            const filteredData = verTimeSeries.filter(item => {
                if (!item) return false;
                const rawDateVal = item.date || item.datetime || item.heure || item.mois || item.annee;
                if (!rawDateVal) return true;

                const dateStr = String(rawDateVal).substring(0, 10);
                if (startDate && dateStr < startDate) return false;
                if (endDate && dateStr > endDate) return false;

                return true;
            });

            if (filteredData.length === 0) return;

            const color = getDynamicColor(index, totalVersions);
            const isSelected = ver === selectedVersion;

            traces.push({
                x: filteredData.map(item => item[timeXAxisKey]),
                y: filteredData.map(item => item.valeur_moyenne),
                type: 'scatter',
                mode: 'lines+markers',
                name: `Version: ${ver}`,
                line: {
                    color: color,
                    width: isSelected ? 3 : 1.5,
                    dash: isSelected ? 'solid' : 'dot'
                },
                marker: { size: isSelected ? 5 : 3, color: color }
            });
        });

        return traces;
    }, [multiVersionTimeSeries, filteredVersionList, selectedVersion, timeGranularity, timeXAxisKey, startDate, endDate]);

    // --- Styles đã căn chỉnh tối ưu chống vỡ giao diện ---
    const styles = {
        wrapper: { minHeight: '100vh', backgroundColor: '#f8fafc', padding: '24px', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1e293b' },
        container: { maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' },
        card: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)', boxSizing: 'border-box' },
        headerCard: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '16px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' },
        topBar: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', flexWrap: 'nowrap' },
        titleSection: { display: 'flex', alignItems: 'center', gap: '16px', minWidth: 0, flexShrink: 1 },
        filterGroup: { display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0 },
        selectBox: { display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '8px 12px', borderRadius: '16px', fontSize: '12px', width: '220px', flexShrink: 0, boxSizing: 'border-box' },
        select: { border: 'none', background: 'transparent', fontWeight: 'bold', color: '#0f172a', outline: 'none', cursor: 'pointer', appearance: 'none', width: '100%', lineHeight: '1', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' },
        kpiGrid: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '12px 16px', display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' },
        kpiItem: { display: 'flex', alignItems: 'center', gap: '12px', padding: '8px 4px', minWidth: 0 },
        iconBg: (bg, color) => ({ backgroundColor: bg, color: color, padding: '10px', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }),
        tabBtn: (isActive) => ({ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', borderRadius: '14px', fontSize: '13px', fontWeight: 'bold', border: '1px solid', borderColor: isActive ? '#2563eb' : '#e2e8f0', cursor: 'pointer', transition: 'all 0.2s', backgroundColor: isActive ? '#2563eb' : '#ffffff', color: isActive ? '#ffffff' : '#64748b' }),
    };

    if (loading && !data) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '500px', gap: '16px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '50%', border: '4px solid #dbeafe', borderTopColor: '#2563eb', animation: 'spin 1s linear infinite' }} />
                <p style={{ color: '#64748b', fontWeight: '500', fontSize: '14px' }}>Chargement du profilage détaillé...</p>
                <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
            </div>
        );
    }

    const stats = data?.statistiques || {};
    const boxPlot = data?.graphiques?.box_plot;
    const histogramme = data?.graphiques?.histogramme || [];

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* EN-TÊTE & FILTRES */}
                <div style={styles.headerCard}>
                    <div style={styles.topBar}>
                        <div style={styles.titleSection}>
                            <div style={styles.iconBg('#eff6ff', '#2563eb')}>
                                <BarChart3 size={24} />
                            </div>
                            <div style={{ minWidth: 0 }}>
                                <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    Profilage Détaillé par Variable
                                </h1>
                                <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', lineHeight: '1.2', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    Variable : <span style={{ color: '#2563eb' }}>{data?.nom_colonne || selectedColumn}</span> | Version : <span style={{ color: '#059669' }}>{selectedVersion}</span>
                                </p>
                            </div>
                        </div>

                        <div style={styles.filterGroup}>
                            {/* Filtre Attraction */}
                            <div style={styles.selectBox}>
                                <Filter size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                                <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>Attraction :</span>
                                <div style={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'center' }}>
                                    <select value={selectedAttraction} onChange={(e) => handleAttractionChange(e.target.value)} style={styles.select}>
                                        <option value="ALL">Toutes les attractions</option>
                                        {attractionList.map(id => <option key={id} value={id}>Attraction {id}</option>)}
                                    </select>
                                </div>
                                <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                            </div>

                            {/* Filtre Colonne */}
                            <div style={styles.selectBox}>
                                <Sliders size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                                <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>Colonne :</span>
                                <div style={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'center' }}>
                                    <select value={selectedColumn} onChange={(e) => setSelectedColumn(e.target.value)} style={styles.select}>
                                        {availableColumns.length > 0 ? (
                                            availableColumns.map(col => <option key={col} value={col}>{col}</option>)
                                        ) : (
                                            <option value={selectedColumn}>{selectedColumn}</option>
                                        )}
                                    </select>
                                </div>
                                <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                            </div>
                        </div>
                    </div>

                    {/* STEPPER CHỌN VERSION CHUẨN ĐẸP */}
                    <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', border: '1px solid #e2e8f0', padding: '16px 20px', width: '100%', boxSizing: 'border-box' }}>
                        <div style={{ fontSize: '13px', fontWeight: '700', color: '#1e293b', marginBottom: '14px' }}>
                            Choisissez la version des données
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between', overflowX: 'auto', paddingBottom: '4px' }}>
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
                                                width: '26px', height: '26px', borderRadius: '50%',
                                                backgroundColor: isSelected ? verColor : '#94a3b8',
                                                color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                fontSize: '11px', fontWeight: '700', transition: 'all 0.2s ease',
                                                boxShadow: isSelected ? `0 0 0 4px ${verColor}25` : 'none', flexShrink: 0
                                            }}>
                                                {index + 1}
                                            </div>
                                            <span style={{ fontSize: '12px', fontWeight: isSelected ? '700' : '500', color: isSelected ? '#1e293b' : '#64748b', whiteSpace: 'nowrap' }}>
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

                {/* KPI METRICS BAR */}
                <div style={styles.kpiGrid}>
                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#faf5ff', '#9333ea')}><Layers size={18} /></div>
                        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Type</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#0f172a', marginTop: '2px', lineHeight: '1.2' }}>{stats.type_donnees || 'N/A'}</div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fffbeb', '#d97706')}><AlertTriangle size={18} /></div>
                        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Manquants</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#d97706', marginTop: '2px', lineHeight: '1.2' }}>
                                {(stats.valeurs_manquantes || 0).toLocaleString()} <span style={{ fontSize: '11px' }}>({stats.pourcentage_manquants || 0}%)</span>
                            </div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#ecfdf5', '#059669')}><Hash size={18} /></div>
                        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Valeurs Zéro</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#059669', marginTop: '2px', lineHeight: '1.2' }}>
                                {(stats.valeurs_zero || 0).toLocaleString()} <span style={{ fontSize: '11px' }}>({stats.pourcentage_zeros || 0}%)</span>
                            </div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fff1f2', '#e11d48')}><TrendingDown size={18} /></div>
                        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Négatives</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#e11d48', marginTop: '2px', lineHeight: '1.2' }}>
                                {(stats.valeurs_negatives || 0).toLocaleString()} <span style={{ fontSize: '11px' }}>({stats.pourcentage_negatifs || 0}%)</span>
                            </div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fef2f2', '#dc2626')}><Activity size={18} /></div>
                        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Outliers</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#dc2626', marginTop: '2px', lineHeight: '1.2' }}>
                                {(stats.valeurs_aberrantes || 0).toLocaleString()} <span style={{ fontSize: '11px' }}>({stats.pourcentage_outliers || 0}%)</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* BOX PLOT CARD */}
                <div style={styles.card}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <BarChart2 size={18} color="#2563eb" />
                            <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a', lineHeight: '1.2' }}>
                                Box Plot & Distribution
                            </h2>
                        </div>
                        {boxPlot && (
                            <div style={{ padding: '6px 12px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px', fontSize: '12px', color: '#334155' }}>
                                <span style={{ fontWeight: 'bold', color: '#0f172a' }}>Intervalle Outliers : </span>
                                <span style={{ fontFamily: 'monospace', fontWeight: '600' }}>[{boxPlot.borne_inf} ; {boxPlot.borne_sup}]</span>
                            </div>
                        )}
                    </div>

                    {boxPlot ? (
                        <div style={{ width: '100%', height: '280px' }}>
                            <Plot
                                data={[{
                                    type: 'box',
                                    q1: [boxPlot.q25],
                                    median: [boxPlot.mediane],
                                    q3: [boxPlot.q75],
                                    lowerfence: [boxPlot.borne_inf],
                                    upperfence: [boxPlot.borne_sup],
                                    marker: { color: '#2563eb' },
                                    name: selectedColumn,
                                    boxpoints: false
                                }]}
                                layout={{
                                    autosize: true,
                                    margin: { t: 20, r: 20, l: 50, b: 50 },
                                    xaxis: { title: { text: selectedColumn, font: { size: 11, color: '#475569', weight: 'bold' } }, automargin: true, showgrid: false },
                                    yaxis: { automargin: true, tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' },
                                    showlegend: false
                                }}
                                useResizeHandler={true}
                                style={{ width: '100%', height: '100%' }}
                                config={{ displayModeBar: false }}
                            />
                        </div>
                    ) : (
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '200px', color: '#94a3b8', fontStyle: 'italic', fontSize: '12px' }}>
                            Non applicable pour ce type de variable.
                        </div>
                    )}
                </div>

                {/* HISTOGRAM CARD */}
                <div style={styles.card}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a', lineHeight: '1.2' }}>
                            Histogramme de Distribution
                        </h2>
                        <span style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '500' }}>Fréquence des valeurs</span>
                    </div>

                    <div style={{ width: '100%', height: '300px' }}>
                        {histogramme.length > 0 ? (
                            <Plot
                                data={[{
                                    x: histogramme.map(item => String(item?.bin_range || '').replace(/\.0\b/g, '').replace(' - ', ' – ')),
                                    y: histogramme.map(item => item.count),
                                    type: 'bar',
                                    text: histogramme.map(item => (item.count > 0 ? item.count.toLocaleString() : '')),
                                    textposition: 'outside',
                                    cliponaxis: false,
                                    marker: { color: '#3b82f6', line: { color: '#1d4ed8', width: 1 } },
                                    hovertemplate: '<b>Intervalle:</b> %{x}<br><b>Fréquence:</b> %{y:,}<extra></extra>'
                                }]}
                                layout={{
                                    autosize: true,
                                    margin: { t: 25, r: 20, l: 50, b: 60 },
                                    bargap: 0.15,
                                    xaxis: { automargin: true, tickangle: -30, tickfont: { size: 10, color: '#64748b' }, showgrid: false },
                                    yaxis: { automargin: true, tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' }
                                }}
                                useResizeHandler={true}
                                style={{ width: '100%', height: '100%' }}
                                config={{ displayModeBar: false }}
                            />
                        ) : (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '200px', color: '#94a3b8', fontStyle: 'italic', fontSize: '12px' }}>
                                Aucune donnée d'histogramme disponible.
                            </div>
                        )}
                    </div>
                </div>

                {/* MULTI-VERSION TIME SERIES CARD */}
                <div style={styles.card}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', borderBottom: '1px solid #f1f5f9', paddingBottom: '16px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <div style={styles.iconBg('#ecfdf5', '#059669')}><Clock size={20} /></div>
                                <div>
                                    <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a', lineHeight: '1.2' }}>
                                        Comparaison des Tendances Temporelles Multi-Versions
                                    </h2>
                                    <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8', lineHeight: '1.2' }}>
                                        Superposition des courbes temporelles pour analyser les écarts entre les versions
                                    </p>
                                </div>
                            </div>

                            {/* Granularity Buttons */}
                            <div style={{ display: 'flex', backgroundColor: '#f1f5f9', padding: '4px', borderRadius: '14px', gap: '4px' }}>
                                {[
                                    { key: 'par_heure', label: 'Par Heure' },
                                    { key: 'par_jour', label: 'Par Jour' },
                                    { key: 'par_mois', label: 'Par Mois' },
                                    { key: 'par_annee', label: 'Par Année' },
                                ].map((btn) => (
                                    <button
                                        key={btn.key}
                                        onClick={() => setTimeGranularity(btn.key)}
                                        style={styles.tabBtn(timeGranularity === btn.key)}
                                    >
                                        {btn.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Date Filter Toolbar */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '12px', flexWrap: 'wrap' }}>
                            <span style={{ fontSize: '12px', fontWeight: '600', color: '#64748b', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <Calendar size={14} color="#94a3b8" /> Période :
                            </span>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '6px 12px', borderRadius: '12px', fontSize: '12px' }}>
                                <span style={{ color: '#94a3b8' }}>Du</span>
                                <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} style={{ border: 'none', background: 'transparent', outline: 'none', fontWeight: 'bold', color: '#1e293b', cursor: 'pointer' }} />
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '6px 12px', borderRadius: '12px', fontSize: '12px' }}>
                                <span style={{ color: '#94a3b8' }}>Au</span>
                                <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} style={{ border: 'none', background: 'transparent', outline: 'none', fontWeight: 'bold', color: '#1e293b', cursor: 'pointer' }} />
                            </div>
                            {(startDate || endDate) && (
                                <button onClick={() => { setStartDate(''); setEndDate(''); }} style={{ padding: '6px 12px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', color: '#dc2626', borderRadius: '12px', fontSize: '12px', fontWeight: 'bold', cursor: 'pointer' }}>
                                    Réinitialiser
                                </button>
                            )}
                        </div>
                    </div>

                    <div style={{ width: '100%', height: '320px', marginTop: '16px' }}>
                        {multiVersionTraces.length > 0 ? (
                            <Plot
                                data={multiVersionTraces}
                                layout={{
                                    autosize: true,
                                    margin: { t: 20, r: 20, l: 50, b: 50 },
                                    xaxis: { tickfont: { size: 10, color: '#64748b' } },
                                    yaxis: { tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' },
                                    legend: { orientation: 'h', y: 1.15, x: 0, font: { size: 11, color: '#334155' } },
                                    hovermode: 'x unified'
                                }}
                                useResizeHandler={true}
                                style={{ width: '100%', height: '100%' }}
                                config={{ displayModeBar: false }}
                            />
                        ) : (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8', fontStyle: 'italic', fontSize: '12px' }}>
                                Aucune donnée temporelle multi-versions disponible pour cette configuration.
                            </div>
                        )}
                    </div>
                </div>

            </div>
        </div>
    );
}