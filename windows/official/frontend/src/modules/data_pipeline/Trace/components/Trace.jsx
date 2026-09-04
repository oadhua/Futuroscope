import React, { useState, useEffect, useMemo } from 'react';
import {
    Activity, Filter, Layers, RefreshCw, AlertTriangle, CheckCircle2,
    BarChart2, Eye, Trash2
} from 'lucide-react';
import Plot from 'react-plotly.js';

export default function PreprocessingTraceabilityModule() {
    // États de sélection
    const [versions, setVersions] = useState({});
    const [selectedVersionId, setSelectedVersionId] = useState('');
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [selectedColumn, setSelectedColumn] = useState('');

    // États des données et des métadonnées
    const [availableColumns, setAvailableColumns] = useState([]);
    const [comparisonData, setComparisonData] = useState(null);

    // États UI
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);

    const API_BASE = 'http://localhost:8000';

    // 1. Chargement de la liste des versions depuis le registre
    const fetchVersions = async () => {
        try {
            const res = await fetch(`${API_BASE}/data-prep/versions`);
            if (res.ok) {
                const data = await res.json();
                setVersions(data || {});
            }
        } catch (err) {
            console.error('Erreur lors du chargement des versions:', err);
        }
    };

    useEffect(() => {
        fetchVersions();
    }, []);

    // 2. Lọc và Sắp xếp danh sách versions theo id_attraction (TĂNG DẦN)
    const availableVersions = useMemo(() => {
        return Object.values(versions)
            .filter(v => {
                if (selectedAttraction === 'ALL') return true;
                const attrVal = v.id_attraction || v.attraction_id;
                if (!attrVal) return true;
                return String(attrVal) === String(selectedAttraction);
            })
            .sort((a, b) => {
                return String(a.version_id).localeCompare(String(b.version_id), undefined, { numeric: true, sensitivity: 'base' });
            });
    }, [versions, selectedAttraction]);

    // Tự động chọn version đầu tiên khi danh sách bị thay đổi do đổi Attraction
    useEffect(() => {
        if (availableVersions.length > 0) {
            const exists = availableVersions.some(v => v.version_id === selectedVersionId);
            if (!exists) {
                setSelectedVersionId(availableVersions[0].version_id);
            }
        } else {
            setSelectedVersionId('');
        }
    }, [availableVersions]);

    // 3. Synchronisation automatique des colonnes de la version sélectionnée
    useEffect(() => {
        if (selectedVersionId && versions[selectedVersionId]) {
            const currentVer = versions[selectedVersionId];

            let cols = [];
            if (currentVer.target_columns) {
                if (Array.isArray(currentVer.target_columns)) {
                    cols = currentVer.target_columns;
                } else if (typeof currentVer.target_columns === 'string') {
                    cols = currentVer.target_columns.split(',').map(c => c.trim()).filter(Boolean);
                }
            } else if (currentVer.stats && currentVer.stats.generated_columns) {
                cols = currentVer.stats.generated_columns;
            }

            setAvailableColumns(cols);
            if (cols.length > 0) {
                setSelectedColumn(cols[0]);
            } else {
                setSelectedColumn('');
            }
        }
    }, [selectedVersionId, versions]);

    // 4. Exécution du rapport comparatif
    const handleFetchTraceability = async () => {
        if (!selectedVersionId) {
            setErrorMsg('Veuillez sélectionner une version de données.');
            return;
        }

        setLoading(true);
        setErrorMsg(null);

        try {
            const queryParams = new URLSearchParams({
                id_attraction: selectedAttraction,
            });
            if (selectedColumn) {
                queryParams.append('column_name', selectedColumn);
            }

            const url = `${API_BASE}/data-prep/compare/${selectedVersionId}?${queryParams.toString()}`;
            const res = await fetch(url);

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Erreur lors de la récupération du rapport de traçabilité.');
            }

            const data = await res.json();
            setComparisonData(data);
        } catch (err) {
            setErrorMsg(err.message);
            setComparisonData(null);
        } finally {
            setLoading(false);
        }
    };

    // useEffect(() => {
    //     if (selectedVersionId) {
    //         handleFetchTraceability();
    //     }
    // }, [selectedVersionId, selectedColumn, selectedAttraction]);

    // 5. Suppression d'une version
    const handleDeleteVersion = async (e, versionId) => {
        e.stopPropagation();
        if (!window.confirm(`Êtes-vous sûr de vouloir supprimer la version ${versionId} ?`)) return;

        try {
            const res = await fetch(`${API_BASE}/data-prep/versions/${versionId}`, { method: 'DELETE' });
            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Impossible de supprimer cette version.');
            }
            if (selectedVersionId === versionId) {
                setSelectedVersionId('');
                setComparisonData(null);
            }
            fetchVersions();
        } catch (err) {
            alert(err.message);
        }
    };

    // Chuẩn bị dữ liệu đồ thị
    const timestamps = comparisonData?.chart_data?.map(d => d.timestamp) || [];
    const origValues = comparisonData?.chart_data?.map(d => d.original_value) || [];
    const procValues = comparisonData?.chart_data?.map(d => d.processed_value) || [];

    const textBreakStyle = {
        wordBreak: 'break-all',
        overflowWrap: 'anywhere',
        whiteSpace: 'normal',
        maxWidth: '100%'
    };

    const styles = {
        wrapper: { minHeight: '100vh', backgroundColor: '#f8fafc', padding: '24px', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1e293b', boxSizing: 'border-box' },
        container: { maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px', width: '100%', boxSizing: 'border-box' },
        card: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)', boxSizing: 'border-box', width: '100%' },

        filterRow: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '16px' },
        formGroup: { width: '100%', boxSizing: 'border-box' },
        label: { display: 'block', fontSize: '12px', fontWeight: '700', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.025em', marginBottom: '8px' },
        selectBox: { display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '8px 14px', borderRadius: '16px', fontSize: '13px', width: '100%', boxSizing: 'border-box' },
        select: { border: 'none', background: 'transparent', fontWeight: 'bold', color: '#0f172a', outline: 'none', cursor: 'pointer', width: '100%', fontSize: '13px', ...textBreakStyle },
        input: { width: '100%', padding: '10px 14px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '14px', fontSize: '13px', fontWeight: '600', color: '#0f172a', outline: 'none', boxSizing: 'border-box' },

        actionBtn: { width: '100%', padding: '12px 18px', backgroundColor: '#2563eb', color: '#ffffff', fontWeight: '700', border: 'none', borderRadius: '16px', cursor: 'pointer', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', transition: 'background-color 0.2s' },

        resultCard: { backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '24px', padding: '24px', width: '100%', boxSizing: 'border-box' },
        tableWrapper: { width: '100%', overflowX: 'auto', borderRadius: '16px', border: '1px solid #e2e8f0', marginBottom: '20px', backgroundColor: '#ffffff' },
        table: { width: '100%', minWidth: '600px', borderCollapse: 'separate', borderSpacing: 0, fontSize: '13px' },
        th: { padding: '12px 16px', textAlign: 'left', color: '#475569', fontWeight: '700', borderBottom: '1px solid #e2e8f0', backgroundColor: '#f8fafc' },
        td: { padding: '12px 16px', borderBottom: '1px solid #f1f5f9', ...textBreakStyle },

        chartStack: { display: 'flex', flexDirection: 'column', gap: '20px', width: '100%', marginTop: '16px' },

        versionItem: (isSelected) => ({
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px', borderRadius: '16px', cursor: 'pointer',
            border: '1px solid', borderColor: isSelected ? '#2563eb' : '#e2e8f0', backgroundColor: isSelected ? '#f0f6ff' : '#ffffff',
            transition: 'all 0.15s ease', gap: '16px', boxSizing: 'border-box', width: '100%'
        })
    };

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* 1. HEADER */}
                <div style={styles.card}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        <div style={{ backgroundColor: '#eff6ff', color: '#2563eb', padding: '12px', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                            <Activity size={28} />
                        </div>
                        <div>
                            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2', ...textBreakStyle }}>
                                Traçabilité et Comparaison des Prétraitements
                            </h1>
                            <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: '#64748b', fontWeight: '600', ...textBreakStyle }}>
                                Analyse comparative globale et détaillée entre les versions d'origine et transformées
                            </p>
                        </div>
                    </div>
                </div>

                {/* 2. PANEL DE CONFIGURATION */}
                <div style={styles.card}>
                    <h2 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: '700', color: '#0f172a', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' }}>
                        ⚙️ Sélection de la Traçabilité
                    </h2>

                    <div style={styles.filterRow}>
                        {/* Périmètre Attraction (Đưa lên làm điều kiện chọn đầu tiên) */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>1. Périmètre Attraction (id_attraction)</label>
                            <div style={{ ...styles.selectBox, backgroundColor: '#f0f9ff', borderColor: '#bae6fd' }}>
                                <Filter size={16} color="#0284c7" style={{ flexShrink: 0 }} />
                                <select 
                                    value={selectedAttraction} 
                                    onChange={(e) => setSelectedAttraction(e.target.value)} 
                                    style={{ ...styles.select, color: '#0369a1' }}
                                >
                                    <option value="ALL">🌐 Tous les sites (ALL)</option>
                                    {['H03', 'H07'].map((attr) => (
                                        <option key={attr} value={attr}>🎢 Attraction ID: {attr}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* Sélection Version (Lọc theo Attraction & Sắp xếp TĂNG DẦN) */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>2. Version à Analyser</label>
                            <div style={styles.selectBox}>
                                <Layers size={16} color="#94a3b8" style={{ flexShrink: 0 }} />
                                <select 
                                    value={selectedVersionId} 
                                    onChange={(e) => setSelectedVersionId(e.target.value)} 
                                    style={styles.select}
                                >
                                    {availableVersions.length === 0 ? (
                                        <option value="">Aucune version disponible</option>
                                    ) : (
                                        availableVersions.map((v) => (
                                            <option key={v.version_id} value={v.version_id}>
                                                {v.version_id}                                            
                                            </option>
                                        ))
                                    )}
                                </select>
                            </div>
                        </div>

                        {/* Colonne Détaillée */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>3. Colonne Analyse Détaillée (Optionnel)</label>
                            <div style={styles.selectBox}>
                                <BarChart2 size={16} color="#94a3b8" style={{ flexShrink: 0 }} />
                                {availableColumns.length > 0 ? (
                                    <select 
                                        value={selectedColumn} 
                                        onChange={(e) => setSelectedColumn(e.target.value)} 
                                        style={styles.select}
                                    >
                                        {availableColumns.map((col) => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                ) : (
                                    <input
                                        type="text"
                                        placeholder="Ex: Gas_01"
                                        value={selectedColumn}
                                        onChange={(e) => setSelectedColumn(e.target.value)}
                                        style={styles.input}
                                    />
                                )}
                            </div>
                        </div>
                    </div>

                    <button 
                        onClick={handleFetchTraceability} 
                        disabled={loading || !selectedVersionId} 
                        style={{ ...styles.actionBtn, backgroundColor: loading ? '#94a3b8' : '#2563eb' }}
                    >
                        {loading ? <RefreshCw className="animate-spin" size={18} /> : <Eye size={18} />}
                        {loading ? 'Génération du Rapport en cours...' : 'Générer le Rapport Comparatif'}
                    </button>

                    {errorMsg && (
                        <div style={{ marginTop: '16px', padding: '14px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '16px', color: '#991b1b', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <AlertTriangle size={18} color="#dc2626" style={{ flexShrink: 0 }} />
                            <span style={textBreakStyle}>{errorMsg}</span>
                        </div>
                    )}
                </div>

                {/* 3. RAPPORT DÉTAILLÉ DE COMPARABILITÉ */}
                {comparisonData ? (
                    <div style={styles.resultCard}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
                            <CheckCircle2 size={22} color="#1e40af" style={{ flexShrink: 0 }} />
                            <h2 style={{ margin: 0, fontWeight: '800', color: '#1e40af', fontSize: '16px', ...textBreakStyle }}>
                                Rapport Comparatif de Traçabilité
                            </h2>
                        </div>

                        {/* BADGES METADATAS */}
                        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '20px' }}>
                            <span style={{ backgroundColor: '#dbeafe', color: '#1e40af', fontFamily: 'monospace', fontSize: '13px', padding: '6px 12px', borderRadius: '10px', fontWeight: 'bold', ...textBreakStyle }}>
                                Version : {comparisonData.version_id}
                            </span>
                            <span style={{ backgroundColor: '#ffffff', color: '#475569', fontFamily: 'monospace', fontSize: '13px', padding: '6px 12px', borderRadius: '10px', fontWeight: 'bold', border: '1px solid #cbd5e1', ...textBreakStyle }}>
                                Parent : {comparisonData.parent_version_id}
                            </span>
                            <span style={{ backgroundColor: '#e0f2fe', color: '#0369a1', fontFamily: 'monospace', fontSize: '13px', padding: '6px 12px', borderRadius: '10px', fontWeight: 'bold', ...textBreakStyle }}>
                                Attraction : {comparisonData.id_attraction}
                            </span>
                        </div>

                        <div style={{ fontSize: '13px', color: '#334155', marginBottom: '20px', display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
                            <div style={textBreakStyle}><strong>Opération :</strong> {comparisonData.operation}</div>
                            <div style={textBreakStyle}><strong>Méthode :</strong> {comparisonData.method}</div>
                            <div style={textBreakStyle}><strong>Colonne :</strong> <code style={textBreakStyle}>{comparisonData.column_name}</code></div>
                        </div>

                        {/* COMPARAISON STATISTIQUES GLOBALES */}
                        <h3 style={{ fontSize: '14px', fontWeight: '800', color: '#0f172a', marginBottom: '10px' }}>
                            📊 Aperçu Global du Jeu de Données
                        </h3>
                        <div style={styles.tableWrapper}>
                            <table style={styles.table}>
                                <thead>
                                    <tr>
                                        <th style={styles.th}>Métrique</th>
                                        <th style={{ ...styles.th, textAlign: 'center' }}>Original ({comparisonData.parent_version_id})</th>
                                        <th style={{ ...styles.th, textAlign: 'center' }}>Transformé ({comparisonData.version_id})</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td style={styles.td}>Lignes Totales</td>
                                        <td style={{ ...styles.td, textAlign: 'center' }}>{comparisonData.original_overview?.num_rows ?? 'N/A'}</td>
                                        <td style={{ ...styles.td, textAlign: 'center', fontWeight: '700' }}>{comparisonData.processed_overview?.num_rows ?? 'N/A'}</td>
                                    </tr>
                                    <tr>
                                        <td style={styles.td}>Colonnes Totales</td>
                                        <td style={{ ...styles.td, textAlign: 'center' }}>{comparisonData.original_overview?.num_columns ?? 'N/A'}</td>
                                        <td style={{ ...styles.td, textAlign: 'center', fontWeight: '700' }}>{comparisonData.processed_overview?.num_columns ?? 'N/A'}</td>
                                    </tr>
                                    <tr>
                                        <td style={styles.td}>Valeurs Manquantes</td>
                                        <td style={{ ...styles.td, textAlign: 'center', color: comparisonData.original_overview?.missing_values > 0 ? '#dc2626' : '#16a34a' }}>
                                            {comparisonData.original_overview?.missing_values ?? 0}
                                        </td>
                                        <td style={{ ...styles.td, textAlign: 'center', color: comparisonData.processed_overview?.missing_values > 0 ? '#dc2626' : '#16a34a', fontWeight: '700' }}>
                                            {comparisonData.processed_overview?.missing_values ?? 0}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style={styles.td}>Valeurs Aberrantes (Outliers)</td>
                                        <td style={{ ...styles.td, textAlign: 'center' }}>{comparisonData.original_overview?.outliers_count ?? 0}</td>
                                        <td style={{ ...styles.td, textAlign: 'center', fontWeight: '700' }}>{comparisonData.processed_overview?.outliers_count ?? 0}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>

                        {/* STATISTIQUES DÉTAILLÉES DE LA COLONNE */}
                        {comparisonData.processed_column_stats && (
                            <>
                                <h3 style={{ fontSize: '14px', fontWeight: '800', color: '#0f172a', marginBottom: '10px' }}>
                                    📈 Comparaison Détaillée : <code style={textBreakStyle}>{comparisonData.column_name}</code>
                                </h3>
                                <div style={styles.tableWrapper}>
                                    <table style={styles.table}>
                                        <thead>
                                            <tr>
                                                <th style={styles.th}>Indicateur</th>
                                                <th style={{ ...styles.th, textAlign: 'center' }}>Original</th>
                                                <th style={{ ...styles.th, textAlign: 'center' }}>Prétraité</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td style={styles.td}>Min / Max</td>
                                                <td style={{ ...styles.td, textAlign: 'center' }}>
                                                    {comparisonData.original_column_stats?.min_val ?? 'N/A'} / {comparisonData.original_column_stats?.max_val ?? 'N/A'}
                                                </td>
                                                <td style={{ ...styles.td, textAlign: 'center', fontWeight: '700' }}>
                                                    {comparisonData.processed_column_stats.min_val ?? 'N/A'} / {comparisonData.processed_column_stats.max_val ?? 'N/A'}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style={styles.td}>Moyenne</td>
                                                <td style={{ ...styles.td, textAlign: 'center' }}>
                                                    {comparisonData.original_column_stats?.mean_val?.toFixed(2) ?? 'N/A'}
                                                </td>
                                                <td style={{ ...styles.td, textAlign: 'center', fontWeight: '700' }}>
                                                    {comparisonData.processed_column_stats.mean_val?.toFixed(2) ?? 'N/A'}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style={styles.td}>Valeurs Uniques</td>
                                                <td style={{ ...styles.td, textAlign: 'center' }}>
                                                    {comparisonData.original_column_stats?.unique_count ?? 'N/A'}
                                                </td>
                                                <td style={{ ...styles.td, textAlign: 'center', fontWeight: '700' }}>
                                                    {comparisonData.processed_column_stats.unique_count ?? 'N/A'}
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </>
                        )}

                        {/* ĐỒ THỊ PLOTLY */}
                        {timestamps.length > 0 && (
                            <>
                                <h3 style={{ fontSize: '14px', fontWeight: '800', color: '#0f172a', marginBottom: '10px' }}>
                                    📉 Visualisation Temporelle
                                </h3>
                                <div style={styles.chartStack}>
                                    <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', padding: '16px', border: '1px solid #e2e8f0', width: '100%', boxSizing: 'border-box' }}>
                                        <div style={{ fontSize: '13px', fontWeight: '700', color: '#2563eb', marginBottom: '8px' }}>🔵 Données Originales</div>
                                        <Plot
                                            data={[{
                                                x: timestamps,
                                                y: origValues,
                                                type: 'scatter',
                                                mode: 'lines',
                                                line: { color: '#2563eb', width: 1.5 },
                                            }]}
                                            layout={{
                                                autosize: true,
                                                height: 250,
                                                margin: { l: 40, r: 20, t: 10, b: 30 },
                                                showlegend: false,
                                            }}
                                            useResizeHandler={true}
                                            style={{ width: '100%', height: '100%' }}
                                        />
                                    </div>
                                    <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', padding: '16px', border: '1px solid #e2e8f0', width: '100%', boxSizing: 'border-box' }}>
                                        <div style={{ fontSize: '13px', fontWeight: '700', color: '#dc2626', marginBottom: '8px' }}>🔴 Données Prétraitées</div>
                                        <Plot
                                            data={[{
                                                x: timestamps,
                                                y: procValues,
                                                type: 'scatter',
                                                mode: 'lines',
                                                line: { color: '#dc2626', width: 1.5 },
                                            }]}
                                            layout={{
                                                autosize: true,
                                                height: 250,
                                                margin: { l: 40, r: 20, t: 10, b: 30 },
                                                showlegend: false,
                                            }}
                                            useResizeHandler={true}
                                            style={{ width: '100%', height: '100%' }}
                                        />
                                    </div>
                                </div>
                            </>
                        )}

                    </div>
                ) : (
                    <div style={{ ...styles.card, borderStyle: 'dashed', textAlign: 'center', color: '#64748b', fontSize: '14px', padding: '32px' }}>
                        💡 <i>Sélectionnez une version ci-dessus pour afficher son analyse de traçabilité.</i>
                    </div>
                )}

                {/* 4. HISTORIQUE DES VERSIONS */}
                <div style={styles.card}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
                        <h2 style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#0f172a' }}>
                            📂 Versions Disponibles pour Analyse
                        </h2>
                        <span style={{ fontSize: '12px', fontWeight: '700', color: '#0369a1', backgroundColor: '#e0f2fe', padding: '4px 10px', borderRadius: '10px' }}>
                            {selectedAttraction === 'ALL' ? 'Tous les sites' : `Attraction: ${selectedAttraction}`}
                        </span>
                    </div>

                    {availableVersions.length === 0 ? (
                        <p style={{ color: '#94a3b8', fontStyle: 'italic', fontSize: '13px', margin: 0 }}>
                            Aucune version enregistrée dans la base de données.
                        </p>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', width: '100%' }}>
                            {availableVersions.map((v) => {
                                const isSelected = selectedVersionId === v.version_id;
                                const attrValue = v.id_attraction;
                                return (
                                    <div key={v.version_id} onClick={() => setSelectedVersionId(v.version_id)} style={styles.versionItem(isSelected)}>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                                <span style={{ fontFamily: 'monospace', fontWeight: '800', color: '#2563eb', fontSize: '14px', ...textBreakStyle }}>
                                                    {v.version_id}
                                                </span>
                                                <span style={{ backgroundColor: '#f1f5f9', color: '#475569', fontSize: '11px', padding: '3px 8px', borderRadius: '6px', fontWeight: '600', ...textBreakStyle }}>
                                                    {v.method_label_fr || v.method || 'Standard'}
                                                </span>
                                                {attrValue && (
                                                    <span style={{ backgroundColor: '#e0f2fe', color: '#0369a1', fontSize: '11px', padding: '3px 8px', borderRadius: '6px', fontWeight: '700', ...textBreakStyle }}>
                                                        {attrValue}
                                                    </span>
                                                )}
                                            </div>
                                            <p style={{ fontSize: '12px', color: '#64748b', margin: '6px 0 0 0', ...textBreakStyle }}>
                                                Parent: <code style={textBreakStyle}>{v.parent_version_id}</code>
                                            </p>
                                        </div>

                                        <button onClick={(e) => handleDeleteVersion(e, v.version_id)} style={{ background: 'none', border: 'none', color: '#dc2626', cursor: 'pointer', padding: '6px', flexShrink: 0 }}>
                                            <Trash2 size={18} />
                                        </button>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

            </div>
        </div>
    );
}