import React, { useState, useEffect, useMemo } from 'react';
import {
    BarChart3, AlertTriangle, Layers, Calendar,
    Filter, ChevronDown, Activity, TrendingDown,
    Clock, Sliders, Hash, BarChart2
} from 'lucide-react';
import Plot from 'react-plotly.js';

export default function SingleColumn({ initialColumn = 'visitor_count' }) {
    // --- States ---
    const [selectedColumn, setSelectedColumn] = useState(initialColumn);
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [timeGranularity, setTimeGranularity] = useState('par_heure');

    const [data, setData] = useState(null);
    const [availableColumns, setAvailableColumns] = useState([]);
    const [availableAttractions, setAvailableAttractions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');

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

                        setSelectedColumn(prevCol => {
                            if (cols.length > 0 && !cols.includes(prevCol)) {
                                return cols[0];
                            }
                            return prevCol;
                        });
                    }

                    if (resGlobal.liste_attractions && availableAttractions.length === 0) {
                        setAvailableAttractions(resGlobal.liste_attractions);
                    }
                }
            } catch (err) {
                console.warn("Méta-données non disponibles via globale:", err);
            }
        };

        fetchMeta();
    }, [selectedAttraction]);

    // --- Fetch Column Details ---
    const fetchColumnDetail = async () => {
        setLoading(true);
        setError(null);
        try {
            let url = `http://localhost:8000/analysis/colonne-detail?col_name=${selectedColumn}&nb_lignes_apercu=100`;
            if (selectedAttraction && selectedAttraction !== 'ALL') {
                url += `&id_attraction=${selectedAttraction}`;
            }

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            const resData = await response.json();
            setData(resData);
        } catch (err) {
            setError(err.message || `Impossible de charger les détails pour la colonne ${selectedColumn}`);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (selectedColumn) {
            fetchColumnDetail();
        }
    }, [selectedColumn, selectedAttraction]);

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

    const filteredTimeSeriesData = useMemo(() => {
        if (!data?.graphiques?.courbes_temporelles?.[timeGranularity]) return [];
        let rawData = data.graphiques.courbes_temporelles[timeGranularity];
        if (!Array.isArray(rawData)) return [];

        return rawData.filter(item => {
            if (!item) return false;
            const rawDateVal = item.date || item.datetime || item.mois || item.annee;
            if (!rawDateVal) return true;

            const dateStr = String(rawDateVal).substring(0, 10);
            if (startDate && dateStr < startDate) return false;
            if (endDate && dateStr > endDate) return false;

            return true;
        });
    }, [data, timeGranularity, startDate, endDate]);

    // --- Unified Internal Styles ---
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
            boxSizing: 'border-box'
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
            maxWidth: '280px',
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
        kpiGrid: {
            backgroundColor: '#ffffff',
            borderRadius: '24px',
            border: '1px solid #e2e8f0',
            padding: '12px 16px',
            display: 'grid',
            gridTemplateColumns: 'repeat(5, minmax(0, 1fr))',
            gap: '12px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
            minHeight: '80px',
            boxSizing: 'border-box'
        },
        kpiItem: { display: 'flex', alignItems: 'center', gap: '12px', padding: '8px 4px', minWidth: 0 },
        iconBg: (bg, color) => ({ backgroundColor: bg, color: color, padding: '10px', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }),
        tabBtn: (isActive) => ({
            padding: '6px 14px',
            borderRadius: '12px',
            fontSize: '12px',
            fontWeight: 'bold',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s',
            backgroundColor: isActive ? '#ffffff' : 'transparent',
            color: isActive ? '#2563eb' : '#64748b',
            boxShadow: isActive ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
        })
    };

    if (loading) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '500px', gap: '16px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '50%', border: '4px solid #dbeafe', borderTopColor: '#2563eb', animation: 'spin 1s linear infinite' }} />
                <p style={{ color: '#64748b', fontWeight: '500', fontSize: '14px' }}>Chargement du profilage détaillé...</p>
                <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div style={{ maxWidth: '500px', margin: '48px auto', padding: '24px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '20px', display: 'flex', alignItems: 'center', gap: '16px', color: '#991b1b' }}>
                <AlertTriangle size={24} color="#dc2626" />
                <div>
                    <h3 style={{ margin: 0, fontWeight: 'bold', fontSize: '15px' }}>Erreur de chargement</h3>
                    <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: '#b91c1c' }}>{error || "Aucune donnée retournée par le serveur."}</p>
                </div>
            </div>
        );
    }

    const stats = data.statistiques || {};
    const boxPlot = data.graphiques?.box_plot;
    const histogramme = data.graphiques?.histogramme || [];

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* 1. TOP HEADER & FILTER CARD */}
                <div style={styles.headerCard}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flex: '1 1 auto', minWidth: 0 }}>
                        <div style={styles.iconBg('#eff6ff', '#2563eb')}>
                            <BarChart3 size={24} />
                        </div>
                        <div style={{ minWidth: 0, overflow: 'hidden' }}>
                            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.02em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                Profilage Détaillé par Variable
                            </h1>
                            <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                Variable sélectionnée : <span style={{ color: '#2563eb' }}>{data.nom_colonne}</span>
                            </p>
                        </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0 }}>
                        {/* Dropdown Colonne */}
                        <div style={styles.selectBox}>
                            <Sliders size={15} color="#94a3b8" style={{ flexShrink: 0 }} />
                            <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>Colonne :</span>
                            <select value={selectedColumn} onChange={(e) => setSelectedColumn(e.target.value)} style={styles.select}>
                                {availableColumns.length > 0 ? (
                                    availableColumns.map(col => <option key={col} value={col}>{col}</option>)
                                ) : (
                                    <option value={data.nom_colonne}>{data.nom_colonne}</option>
                                )}
                            </select>
                            <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                        </div>

                        {/* Dropdown Attraction */}
                        <div style={styles.selectBox}>
                            <Filter size={15} color="#94a3b8" style={{ flexShrink: 0 }} />
                            <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>Attraction :</span>
                            <select value={selectedAttraction} onChange={(e) => setSelectedAttraction(e.target.value)} style={styles.select}>
                                <option value="ALL">Toutes les attractions</option>
                                {availableAttractions.map(id => <option key={id} value={id}>Attraction {id}</option>)}
                            </select>
                            <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                        </div>
                    </div>
                </div>

                {/* 2. KPI QUALITY METRICS BAR */}
                <div style={styles.kpiGrid}>
                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#faf5ff', '#9333ea')}><Layers size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Type</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#0f172a', marginTop: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{stats.type_donnees || 'N/A'}</div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fffbeb', '#d97706')}><AlertTriangle size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Manquants</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#d97706', marginTop: '2px', whiteSpace: 'nowrap' }}>
                                {(stats.valeurs_manquantes || 0).toLocaleString()} <span style={{ fontSize: '11px', color: '#b45309' }}>({stats.pourcentage_manquants || 0}%)</span>
                            </div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#ecfdf5', '#059669')}><Hash size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Valeurs Zéro</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#059669', marginTop: '2px', whiteSpace: 'nowrap' }}>
                                {(stats.valeurs_zero || 0).toLocaleString()} <span style={{ fontSize: '11px', color: '#047857' }}>({stats.pourcentage_zeros || 0}%)</span>
                            </div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fff1f2', '#e11d48')}><TrendingDown size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Négatives</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#e11d48', marginTop: '2px', whiteSpace: 'nowrap' }}>
                                {(stats.valeurs_negatives || 0).toLocaleString()} <span style={{ fontSize: '11px', color: '#be123c' }}>({stats.pourcentage_negatifs || 0}%)</span>
                            </div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fef2f2', '#dc2626')}><Activity size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Outliers</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#dc2626', marginTop: '2px', whiteSpace: 'nowrap' }}>
                                {(stats.valeurs_aberrantes || 0).toLocaleString()} <span style={{ fontSize: '11px', color: '#b91c1c' }}>({stats.pourcentage_outliers || 0}%)</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* 3. CHARTS ROW: BOXPLOT & HISTOGRAM (Xếp chồng dọc) */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

                    {/* 1. Box Plot Card */}
                    <div style={{ ...styles.card, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: '340px' }}>
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                                <BarChart2 size={18} color="#2563eb" />
                                <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a' }}>Box Plot & Distribution</h2>
                            </div>

                            {boxPlot ? (
                                <div style={{ width: '100%', height: '350px' }}>
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
                                            xaxis: {
                                                title: { text: selectedColumn, font: { size: 11, color: '#475569', weight: 'bold' } },
                                                automargin: true,
                                                showgrid: false
                                            },
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

                        {boxPlot && (
                            <div style={{ padding: '10px 12px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px', fontSize: '12px', color: '#334155', marginTop: '12px' }}>
                                <span style={{ fontWeight: 'bold', color: '#0f172a', display: 'block', marginBottom: '2px' }}>Intervalle Outliers :</span>
                                <span style={{ fontFamily: 'monospace', fontWeight: '600' }}>[{boxPlot.borne_inf} ; {boxPlot.borne_sup}]</span>
                            </div>
                        )}
                    </div>

                    {/* 2. Histogram Card */}
                    <div style={{ ...styles.card, display: 'flex', flexDirection: 'column', minHeight: '380px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                            <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a' }}>Histogramme de Distribution</h2>
                            <span style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '500' }}>Fréquence des valeurs</span>
                        </div>

                        <div style={{ width: '100%', height: '350px' }}>
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
                                        margin: { t: 25, r: 20, l: 50, b: 80 },
                                        bargap: 0.15,
                                        xaxis: { automargin: true, tickangle: -40, tickfont: { size: 10, color: '#64748b' }, showgrid: false },
                                        yaxis: { automargin: true, tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' }
                                    }}
                                    useResizeHandler={true}
                                    style={{ width: '100%', height: '100%' }}
                                    config={{ displayModeBar: false }}
                                />
                            ) : (
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8', fontStyle: 'italic', fontSize: '12px' }}>
                                    Aucune donnée d'histogramme disponible.
                                </div>
                            )}
                        </div>
                    </div>

                </div>

                {/* 4. TIME SERIES TREND CHART */}
                <div style={{ ...styles.card, minHeight: '420px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', borderBottom: '1px solid #f1f5f9', paddingBottom: '16px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <div style={styles.iconBg('#ecfdf5', '#059669')}>
                                    <Clock size={20} />
                                </div>
                                <div>
                                    <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a' }}>Tendance Temporelle Moyenne</h2>
                                    <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>Évolution de la valeur selon la granularité et la période</p>
                                </div>
                            </div>

                            {/* Granularity Buttons */}
                            <div style={{ display: 'flex', backgroundColor: '#f1f5f9', padding: '4px', borderRadius: '16px', gap: '4px' }}>
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
                                <Calendar size={14} color="#94a3b8" />
                                Période :
                            </span>

                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '6px 12px', borderRadius: '12px', fontSize: '12px' }}>
                                <span style={{ color: '#94a3b8' }}>Du</span>
                                <input
                                    type="date"
                                    value={startDate}
                                    onChange={(e) => setStartDate(e.target.value)}
                                    style={{ border: 'none', background: 'transparent', outline: 'none', fontWeight: 'bold', color: '#1e293b', cursor: 'pointer' }}
                                />
                            </div>

                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '6px 12px', borderRadius: '12px', fontSize: '12px' }}>
                                <span style={{ color: '#94a3b8' }}>Au</span>
                                <input
                                    type="date"
                                    value={endDate}
                                    onChange={(e) => setEndDate(e.target.value)}
                                    style={{ border: 'none', background: 'transparent', outline: 'none', fontWeight: 'bold', color: '#1e293b', cursor: 'pointer' }}
                                />
                            </div>

                            {(startDate || endDate) && (
                                <button
                                    onClick={() => { setStartDate(''); setEndDate(''); }}
                                    style={{ padding: '6px 12px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', color: '#dc2626', borderRadius: '12px', fontSize: '12px', fontWeight: 'bold', cursor: 'pointer' }}
                                >
                                    Réinitialiser
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Time Series Plotly Chart */}
                    <div style={{ width: '100%', height: '320px', marginTop: '16px' }}>
                        {filteredTimeSeriesData.length > 0 ? (
                            <Plot
                                data={[{
                                    x: filteredTimeSeriesData.map(item => item[timeXAxisKey]),
                                    y: filteredTimeSeriesData.map(item => item.valeur_moyenne),
                                    type: 'scatter',
                                    mode: 'lines+markers',
                                    line: { color: '#059669', width: 2.5 },
                                    marker: { size: 5, color: '#059669' }
                                }]}
                                layout={{
                                    autosize: true,
                                    margin: { t: 20, r: 20, l: 50, b: 50 },
                                    xaxis: {
                                        tickfont: { size: 10, color: '#64748b' },
                                        dtick: timeGranularity === 'par_annee' ? 1 : undefined,
                                        tickformat: timeGranularity === 'par_annee' ? 'd' : undefined
                                    },
                                    yaxis: { tickfont: { size: 10, color: '#64748b' }, gridcolor: '#f1f5f9' }
                                }}
                                useResizeHandler={true}
                                style={{ width: '100%', height: '100%' }}
                                config={{ displayModeBar: false }}
                            />
                        ) : (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8', fontStyle: 'italic', fontSize: '12px' }}>
                                Aucune donnée temporelle disponible pour cette période ou granularité.
                            </div>
                        )}
                    </div>
                </div>

            </div>
        </div>
    );
}