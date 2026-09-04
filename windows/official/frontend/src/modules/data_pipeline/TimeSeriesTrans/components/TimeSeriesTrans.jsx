import React, { useState, useEffect, useMemo } from 'react';
import {
    TrendingUp, Filter, Layers, CheckSquare, Square,
    Play, Trash2, AlertTriangle, CheckCircle2, X, RefreshCw, Calendar
} from 'lucide-react';

const TRANSFORM_METHODS = [
    {
        value: 'first_diff',
        label: 'Différenciation première',
        description: 'Soustraire chaque observation de la série par son observation précédente (t - t-1)'
    },
    {
        value: 'seasonal_diff',
        label: 'Différenciation saisonnière',
        description: 'Supprimer les tendances et variations saisonnières afin de rendre la moyenne et la variance constantes (t - t-s)'
    },
    {
        value: 'log_transform',
        label: 'Différenciation logarithmique',
        description: 'Appliquer le logarithme aux valeurs de la série temporelle'
    }
];

export default function TimeSeriesTransformModule() {
    const [versions, setVersions] = useState({});
    const [selectedParent, setSelectedParent] = useState('v0_raw');
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [availableAttractions, setAvailableAttractions] = useState([]);

    const [numericColumns, setNumericColumns] = useState([]);
    const [selectedColumns, setSelectedColumns] = useState([]);

    // Paramètres Time Series
    const [method, setMethod] = useState('first_diff');
    const [seasonalPeriod, setSeasonalPeriod] = useState(24);
    const [dropNa, setDropNa] = useState(true);

    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);
    const [selectedVersion, setSelectedVersion] = useState(null);

    const API_BASE = 'http://localhost:8000';

    // 1. Fetch danh sách version từ API
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

    // 2. Tự động đồng bộ id_attraction
    useEffect(() => {
        if (selectedParent && selectedParent !== 'v0_raw' && versions[selectedParent]) {
            const parentObj = versions[selectedParent];
            const parentAttr = parentObj.id_attraction || parentObj.attraction_id;
            if (parentAttr) {
                setSelectedAttraction(String(parentAttr));
                return;
            }
        }

        const match = selectedParent.match(/_(H\d+)$/i);
        if (match) {
            setSelectedAttraction(match[1].toUpperCase());
        } else {
            setSelectedAttraction('ALL');
        }
    }, [selectedParent, versions]);

    // 3. Lấy thông tin cột số từ API Time Series
    useEffect(() => {
        const fetchColumnsAndStats = async () => {
            try {
                const queryParams = new URLSearchParams({
                    id_attraction: selectedAttraction
                });

                const url = `${API_BASE}/data-prep/time-series-transform/versions/${selectedParent}/stats?${queryParams.toString()}`;
                const res = await fetch(url);

                if (res.ok) {
                    const data = await res.json();
                    const numCols = data.numeric_columns || [];
                    setNumericColumns(numCols);

                    if (numCols.length > 0) {
                        setSelectedColumns(prev => prev.length === 0 ? numCols : prev.filter(c => numCols.includes(c)));
                    }

                    const attrs = data.available_attractions || [];
                    if (Array.isArray(attrs) && attrs.length > 0) {
                        setAvailableAttractions(attrs);
                    }
                }
            } catch (err) {
                console.error('Erreur lors du chargement des colonnes:', err);
            }
        };

        if (selectedParent) {
            fetchColumnsAndStats();
        }
    }, [selectedParent, selectedAttraction]);

    const filteredVersions = useMemo(() => {
        return Object.values(versions).filter(v => {
            if (selectedAttraction === 'ALL') return true;
            const attrVal = v.id_attraction || v.attraction_id;
            if (!attrVal) return true;
            return String(attrVal) === String(selectedAttraction);
        });
    }, [versions, selectedAttraction]);

    const handleToggleColumn = (col) => {
        setSelectedColumns(prev =>
            prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
        );
    };

    const handleSelectAllColumns = () => setSelectedColumns([...numericColumns]);
    const handleDeselectAllColumns = () => setSelectedColumns([]);

    // 4. Thực thi Transformation
    const handleExecute = async () => {
        if (selectedColumns.length === 0) {
            setErrorMsg("Veuillez sélectionner au moins une colonne numérique à transformer.");
            return;
        }

        setLoading(true);
        setErrorMsg(null);

        const payload = {
            parent_version_id: selectedParent,
            id_attraction: selectedAttraction,
            target_columns: selectedColumns,
            method: method,
            seasonal_period: parseInt(seasonalPeriod, 10) || 24,
            drop_na: dropNa
        };

        try {
            const res = await fetch(`${API_BASE}/data-prep/time-series-transform`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Une erreur est survenue lors de la transformation.');
            }

            const resultData = await res.json();
            setSelectedVersion(resultData);
            fetchVersions();
        } catch (err) {
            setErrorMsg(err.message);
        } finally {
            setLoading(false);
        }
    };

    // 5. Xóa Version
    const handleDeleteVersion = async (e, versionId) => {
        e.stopPropagation();
        if (!window.confirm(`Êtes-vous sûr de vouloir supprimer la version ${versionId} ?`)) return;

        try {
            const res = await fetch(`${API_BASE}/data-prep/versions/${versionId}`, { method: 'DELETE' });
            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Impossible de supprimer cette version.');
            }
            if (selectedVersion && selectedVersion.version_id === versionId) {
                setSelectedVersion(null);
            }
            fetchVersions();
        } catch (err) {
            alert(err.message);
        }
    };

    const styles = {
        wrapper: { minHeight: '100vh', backgroundColor: '#f8fafc', padding: '24px', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1e293b' },
        container: { maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' },
        card: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)', boxSizing: 'border-box' },
        headerCard: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '16px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' },
        topBar: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' },
        iconBg: (bg, color) => ({ backgroundColor: bg, color: color, padding: '10px', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }),
        gridContainer: { display: 'grid', gridTemplateColumns: 'minmax(340px, 1fr) minmax(420px, 1.4fr)', gap: '20px', alignItems: 'start' },

        formGroup: { marginBottom: '18px' },
        label: { display: 'block', fontSize: '12px', fontWeight: '700', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.025em', marginBottom: '8px' },
        selectBox: { display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '8px 14px', borderRadius: '16px', fontSize: '13px', width: '100%', boxSizing: 'border-box' },
        select: { border: 'none', background: 'transparent', fontWeight: 'bold', color: '#0f172a', outline: 'none', cursor: 'pointer', width: '100%', fontSize: '13px' },
        input: { border: 'none', background: 'transparent', fontWeight: 'bold', color: '#0f172a', outline: 'none', width: '100%', fontSize: '13px' },

        checkboxContainer: {
            maxHeight: '200px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '8px', backgroundColor: '#f8fafc', display: 'flex', flexDirection: 'column', gap: '6px'
        },
        checkboxItem: (isChecked) => ({
            display: 'flex', alignItems: 'center', padding: '8px 12px', borderRadius: '12px', cursor: 'pointer', fontSize: '13px',
            border: '1px solid', borderColor: isChecked ? '#8b5cf6' : 'transparent', backgroundColor: isChecked ? '#f5f3ff' : '#ffffff',
            transition: 'all 0.15s ease', gap: '8px'
        }),

        actionBtn: { width: '100%', padding: '12px 18px', backgroundColor: '#8b5cf6', color: '#ffffff', fontWeight: '700', border: 'none', borderRadius: '16px', cursor: 'pointer', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', transition: 'background-color 0.2s', marginTop: '8px' },
        smallBtn: { padding: '4px 8px', fontSize: '11px', fontWeight: 'bold', backgroundColor: '#ffffff', color: '#475569', border: '1px solid #e2e8f0', borderRadius: '8px', cursor: 'pointer' },

        resultCard: { backgroundColor: '#f5f3ff', border: '1px solid #ddd6fe', borderRadius: '24px', padding: '20px', minWidth: 0 },
        versionItem: (isSelected) => ({
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px', borderRadius: '16px', cursor: 'pointer',
            border: '1px solid', borderColor: isSelected ? '#8b5cf6' : '#e2e8f0', backgroundColor: isSelected ? '#f5f3ff' : '#ffffff',
            transition: 'all 0.15s ease', gap: '12px'
        })
    };

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* HEADER */}
                <div style={styles.headerCard}>
                    <div style={styles.topBar}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={styles.iconBg('#f5f3ff', '#8b5cf6')}>
                                <TrendingUp size={24} />
                            </div>
                            <div>
                                <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2' }}>
                                    Transformation des Séries Temporelles
                                </h1>
                                <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', lineHeight: '1.2' }}>
                                    Stationnarisation et transformation (Différenciation première, saisonnière, et logarithmique)
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* GRID CONTENT */}
                <div style={styles.gridContainer}>

                    {/* PANEL CONFIGURATION */}
                    <div style={styles.card}>
                        <h2 style={{ margin: '0 0 16px 0', fontSize: '15px', fontWeight: '700', color: '#0f172a', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' }}>
                            ⚙️ Configuration de la Transformation
                        </h2>

                        {/* 1. Version Source */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>1. Version Source (Parent)</label>
                            <div style={styles.selectBox}>
                                <Layers size={14} color="#94a3b8" />
                                <select value={selectedParent} onChange={(e) => setSelectedParent(e.target.value)} style={styles.select}>
                                    <option value="v0_raw">v0_raw (Données brutes DB)</option>
                                    {Object.keys(versions).map((vId) => (
                                        <option key={vId} value={vId}>{vId}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* 2. Périmètre Attraction */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>2. Périmètre Attraction (id_attraction)</label>
                            <div style={{ ...styles.selectBox, backgroundColor: '#f0f9ff', borderColor: '#bae6fd' }}>
                                <Filter size={14} color="#0284c7" />
                                <select value={selectedAttraction} onChange={(e) => setSelectedAttraction(e.target.value)} style={{ ...styles.select, color: '#0369a1' }}>
                                    <option value="ALL">🌐 Tous les sites (ALL)</option>
                                    {availableAttractions.map((attr) => (
                                        <option key={attr} value={attr}>🎢 Attraction ID: {attr}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* 3. Choix de la Méthode */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>3. Méthode de Transformation</label>
                            <div style={styles.selectBox}>
                                <select value={method} onChange={(e) => setMethod(e.target.value)} style={styles.select}>
                                    {TRANSFORM_METHODS.map((m) => (
                                        <option key={m.value} value={m.value}>{m.label}</option>
                                    ))}
                                </select>
                            </div>
                            <p style={{ fontSize: '11px', color: '#64748b', marginTop: '6px', fontStyle: 'italic' }}>
                                {TRANSFORM_METHODS.find(m => m.value === method)?.description}
                            </p>
                        </div>

                        {/* Parameter supplémentaire si Seasonal Diff */}
                        {method === 'seasonal_diff' && (
                            <div style={styles.formGroup}>
                                <label style={styles.label}>Période Saisonnière (s)</label>
                                <div style={styles.selectBox}>
                                    <Calendar size={14} color="#8b5cf6" />
                                    <input
                                        type="number"
                                        value={seasonalPeriod}
                                        onChange={(e) => setSeasonalPeriod(e.target.value)}
                                        placeholder="Ex: 24 pour 24 pas de temps"
                                        style={styles.input}
                                    />
                                </div>
                            </div>
                        )}

                        {/* 4. Selection des Colonnes Numériques */}
                        <div style={styles.formGroup}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                <label style={{ ...styles.label, margin: 0 }}>4. Colonnes à Transformer ({selectedColumns.length}/{numericColumns.length})</label>
                                <div style={{ display: 'flex', gap: '4px' }}>
                                    <button type="button" onClick={handleSelectAllColumns} style={styles.smallBtn}>Tout</button>
                                    <button type="button" onClick={handleDeselectAllColumns} style={styles.smallBtn}>Aucun</button>
                                </div>
                            </div>

                            <div style={styles.checkboxContainer}>
                                {numericColumns.map((col) => {
                                    const isChecked = selectedColumns.includes(col);
                                    return (
                                        <div key={col} onClick={() => handleToggleColumn(col)} style={styles.checkboxItem(isChecked)}>
                                            {isChecked ? <CheckSquare size={16} color="#8b5cf6" /> : <Square size={16} color="#94a3b8" />}
                                            <span style={{ fontWeight: isChecked ? '700' : '500', color: '#1e293b', wordBreak: 'break-all', flex: 1 }}>
                                                {col}
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Checkbox Supprimer NaN */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '18px' }}>
                            <input
                                type="checkbox"
                                id="dropNaCheck"
                                checked={dropNa}
                                onChange={(e) => setDropNa(e.target.checked)}
                                style={{ cursor: 'pointer' }}
                            />
                            <label htmlFor="dropNaCheck" style={{ fontSize: '12px', fontWeight: '600', color: '#475569', cursor: 'pointer' }}>
                                Supprimer automatiquement les valeurs NaN créées par le décalage
                            </label>
                        </div>

                        <button onClick={handleExecute} disabled={loading} style={{ ...styles.actionBtn, backgroundColor: loading ? '#94a3b8' : '#8b5cf6' }}>
                            {loading ? <RefreshCw className="animate-spin" size={16} /> : <Play size={16} />}
                            {loading ? 'Calcul en cours...' : 'Exécuter la Transformation'}
                        </button>

                        {errorMsg && (
                            <div style={{ marginTop: '14px', padding: '12px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '14px', color: '#991b1b', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <AlertTriangle size={16} color="#dc2626" />
                                <span>{errorMsg}</span>
                            </div>
                        )}
                    </div>

                    {/* PANEL DROIT: RAPPORT & VERSIONS */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', minWidth: 0 }}>

                        {selectedVersion ? (
                            <div style={styles.resultCard}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <CheckCircle2 size={18} color="#6d28d9" />
                                        <span style={{ fontWeight: '800', color: '#6d28d9', fontSize: '14px' }}>
                                            Rapport de Transformation Temporelle
                                        </span>
                                    </div>
                                    <button onClick={() => setSelectedVersion(null)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}>
                                        <X size={18} />
                                    </button>
                                </div>

                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
                                    <span style={{ backgroundColor: '#ddd6fe', color: '#5b21b6', fontFamily: 'monospace', fontSize: '12px', padding: '4px 8px', borderRadius: '8px', fontWeight: 'bold' }}>
                                        {selectedVersion.version_id}
                                    </span>
                                    <span style={{ backgroundColor: '#e0f2fe', color: '#0369a1', fontFamily: 'monospace', fontSize: '12px', padding: '4px 8px', borderRadius: '8px', fontWeight: 'bold' }}>
                                        Attraction: {selectedVersion.id_attraction || 'ALL'}
                                    </span>
                                </div>

                                <div style={{ fontSize: '12px', color: '#334155', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                    <div><strong>Méthode :</strong> {selectedVersion.method_label_fr || selectedVersion.method}</div>
                                    <div><strong>Version Parent :</strong> <code>{selectedVersion.parent_version_id}</code></div>
                                    <div><strong>Colonnes Transformées :</strong> {selectedVersion.target_columns?.join(', ')}</div>
                                    <div><strong>Lignes Avant :</strong> {selectedVersion.stats?.rows_before}</div>
                                    <div><strong>Lignes Après :</strong> {selectedVersion.stats?.rows_after}</div>
                                    <div style={{ color: '#6d28d9', fontWeight: '700' }}>
                                        <strong>Lignes NaN Supprimées :</strong> {selectedVersion.stats?.nan_rows_dropped}
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div style={{ ...styles.card, borderStyle: 'dashed', textAlign: 'center', color: '#64748b', fontSize: '13px', padding: '24px' }}>
                                💡 <i>Sélectionnez une version ci-dessous pour consulter son rapport PostgreSQL.</i>
                            </div>
                        )}

                        {/* LISTE DES VERSIONS */}
                        <div style={styles.card}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' }}>
                                <h2 style={{ margin: 0, fontSize: '15px', fontWeight: '700', color: '#0f172a' }}>
                                    📂 Versions Enregistrées dans Registry
                                </h2>
                                <span style={{ fontSize: '11px', fontWeight: '700', color: '#0369a1', backgroundColor: '#e0f2fe', padding: '2px 8px', borderRadius: '10px' }}>
                                    {selectedAttraction === 'ALL' ? 'Tous les sites' : `Attraction: ${selectedAttraction}`}
                                </span>
                            </div>

                            {filteredVersions.length === 0 ? (
                                <p style={{ color: '#94a3b8', fontStyle: 'italic', fontSize: '13px', margin: 0 }}>
                                    Aucune version enregistrée.
                                </p>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                    {filteredVersions.map((v) => {
                                        const isSelected = selectedVersion && selectedVersion.version_id === v.version_id;
                                        return (
                                            <div key={v.version_id} onClick={() => setSelectedVersion(v)} style={styles.versionItem(isSelected)}>
                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                                                        <span style={{ fontFamily: 'monospace', fontWeight: '800', color: '#8b5cf6', fontSize: '13px' }}>
                                                            {v.version_id}
                                                        </span>
                                                        <span style={{ backgroundColor: '#f1f5f9', color: '#475569', fontSize: '11px', padding: '2px 6px', borderRadius: '6px', fontWeight: '600' }}>
                                                            {v.method_label_fr || v.method}
                                                        </span>
                                                    </div>
                                                    <p style={{ fontSize: '11px', color: '#64748b', margin: '4px 0 0 0' }}>
                                                        Parent: <code>{v.parent_version_id}</code>
                                                    </p>
                                                </div>

                                                <button onClick={(e) => handleDeleteVersion(e, v.version_id)} style={{ background: 'none', border: 'none', color: '#dc2626', cursor: 'pointer', padding: '4px' }}>
                                                    <Trash2 size={16} />
                                                </button>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>

                    </div>
                </div>

            </div>
        </div>
    );
}