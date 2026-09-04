import React, { useState, useEffect, useMemo } from 'react';
import {
    Sliders, Filter, Layers, CheckSquare, Square,
    Play, Trash2, AlertTriangle, CheckCircle2, X, RefreshCw
} from 'lucide-react';

const SCALING_METHODS = [
    { value: 'standard', label: 'Standard Scaling (Z-score)' },
    { value: 'min_max', label: 'Min-Max Scaling (Plage définie)' },
    { value: 'robust', label: 'Robust Scaling (Médiane & IQR)' },
];

export default function ScalingModule() {
    const [versions, setVersions] = useState({});
    const [selectedParent, setSelectedParent] = useState('v0_raw');
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [availableAttractions, setAvailableAttractions] = useState([]);

    const [availableColumns, setAvailableColumns] = useState([]);
    const [selectedColumns, setSelectedColumns] = useState([]);

    // Paramètres Scaling
    const [selectedMethod, setSelectedMethod] = useState('standard');
    const [minRange, setMinRange] = useState(0.0);
    const [maxRange, setMaxRange] = useState(1.0);

    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);
    const [selectedVersion, setSelectedVersion] = useState(null);

    const API_BASE = 'http://localhost:8000';

    // 1. Fetch danh sách các version trong Registry
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

    // 2. Tự động đồng bộ id_attraction từ version parent
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

    // 3. Lấy trực tiếp danh sách các cột khả dụng và attractions từ API Stats của Scaling Router
    useEffect(() => {
        const fetchColumnsAndStats = async () => {
            try {
                const queryParams = new URLSearchParams({
                    id_attraction: selectedAttraction
                });

                // Gọi đúng endpoint stats mới cập nhật trong scaling_router
                const url = `${API_BASE}/data-prep/scaling/versions/${selectedParent}/stats?${queryParams.toString()}`;
                const res = await fetch(url);

                if (res.ok) {
                    const data = await res.json();

                    // Cập nhật danh sách các cột từ backend trả về
                    const cols = data.available_columns || [];
                    setAvailableColumns(cols);

                    if (cols.length > 0) {
                        setSelectedColumns(prev => prev.length === 0 ? cols : prev);
                    }

                    // Cập nhật danh sách attractions
                    const attrs = data.available_attractions || [];
                    if (Array.isArray(attrs) && attrs.length > 0) {
                        setAvailableAttractions(attrs);
                    }
                }
            } catch (err) {
                console.error('Erreur lors du chargement des colonnes depuis la DB:', err);
            }
        };

        if (selectedParent) {
            fetchColumnsAndStats();
        }
    }, [selectedParent, selectedAttraction]);

    // Lọc danh sách version hiển thị theo attraction được chọn
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

    const handleSelectAllColumns = () => setSelectedColumns([...availableColumns]);
    const handleDeselectAllColumns = () => setSelectedColumns([]);

    // 4. Thực thi Scaling và lưu version vào PostgreSQL
    const handleExecute = async () => {
        if (selectedColumns.length === 0) {
            setErrorMsg('Veuillez sélectionner au moins une colonne à normaliser.');
            return;
        }

        setLoading(true);
        setErrorMsg(null);

        const payload = {
            parent_version_id: selectedParent,
            id_attraction: selectedAttraction,
            target_columns: selectedColumns,
            method: selectedMethod,
            feature_range: selectedMethod === 'min_max' ? [parseFloat(minRange), parseFloat(maxRange)] : [0.0, 1.0],
        };

        try {
            const res = await fetch(`${API_BASE}/data-prep/scaling`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Une erreur est survenue lors de la normalisation.');
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

    // 5. Xóa version khỏi PostgreSQL
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
        input: { width: '100%', padding: '10px 14px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '14px', fontSize: '13px', fontWeight: '600', color: '#0f172a', outline: 'none', boxSizing: 'border-box' },

        checkboxContainer: {
            maxHeight: '220px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '8px', backgroundColor: '#f8fafc', display: 'flex', flexDirection: 'column', gap: '6px'
        },
        checkboxItem: (isChecked) => ({
            display: 'flex', alignItems: 'center', padding: '8px 12px', borderRadius: '12px', cursor: 'pointer', fontSize: '13px',
            border: '1px solid', borderColor: isChecked ? '#93c5fd' : 'transparent', backgroundColor: isChecked ? '#eff6ff' : '#ffffff',
            transition: 'all 0.15s ease', gap: '8px'
        }),

        actionBtn: { width: '100%', padding: '12px 18px', backgroundColor: '#2563eb', color: '#ffffff', fontWeight: '700', border: 'none', borderRadius: '16px', cursor: 'pointer', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', transition: 'background-color 0.2s', marginTop: '8px' },
        smallBtn: { padding: '4px 8px', fontSize: '11px', fontWeight: 'bold', backgroundColor: '#ffffff', color: '#475569', border: '1px solid #e2e8f0', borderRadius: '8px', cursor: 'pointer' },

        resultCard: { backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '24px', padding: '20px', minWidth: 0 },
        table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0, backgroundColor: '#ffffff', borderRadius: '16px', overflow: 'hidden', fontSize: '13px', border: '1px solid #e2e8f0' },
        th: { padding: '10px 14px', textAlign: 'left', color: '#475569', fontWeight: '700', borderBottom: '1px solid #e2e8f0', backgroundColor: '#f8fafc' },
        td: { padding: '10px 14px', borderBottom: '1px solid #f1f5f9' },

        versionItem: (isSelected) => ({
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px', borderRadius: '16px', cursor: 'pointer',
            border: '1px solid', borderColor: isSelected ? '#2563eb' : '#e2e8f0', backgroundColor: isSelected ? '#eff6ff' : '#ffffff',
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
                            <div style={styles.iconBg('#eff6ff', '#2563eb')}>
                                <Sliders size={24} />
                            </div>
                            <div>
                                <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2' }}>
                                    Normalisation & Mise à l'Échelle (Scaling)
                                </h1>
                                <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', lineHeight: '1.2' }}>
                                    Standardisation, mise à l'échelle Min-Max et transformation robuste des caractéristiques numériques
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* GRID CONTENT */}
                <div style={styles.gridContainer}>

                    {/* PANEL DE CONFIGURATION */}
                    <div style={styles.card}>
                        <h2 style={{ margin: '0 0 16px 0', fontSize: '15px', fontWeight: '700', color: '#0f172a', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' }}>
                            ⚙️ Configuration du Scaling
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
                                    {availableAttractions.length > 0 ? (
                                        availableAttractions.map((attr) => (
                                            <option key={attr} value={attr}>🎢 Attraction ID: {attr}</option>
                                        ))
                                    ) : (
                                        ['H03', 'H07'].map((attr) => (
                                            <option key={attr} value={attr}>🎢 Attraction ID: {attr}</option>
                                        ))
                                    )}
                                </select>
                            </div>
                        </div>

                        {/* 3. Selection des Colonnes cibles */}
                        <div style={styles.formGroup}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                <label style={{ ...styles.label, margin: 0 }}>3. Colonnes Cibles ({selectedColumns.length})</label>
                                <div style={{ display: 'flex', gap: '4px' }}>
                                    <button type="button" onClick={handleSelectAllColumns} style={styles.smallBtn}>Tout</button>
                                    <button type="button" onClick={handleDeselectAllColumns} style={styles.smallBtn}>Aucun</button>
                                </div>
                            </div>

                            <div style={styles.checkboxContainer}>
                                {availableColumns.length === 0 ? (
                                    <div style={{ fontSize: '12px', color: '#94a3b8', padding: '8px', textAlign: 'center' }}>
                                        Aucune colonne disponible dans cette table
                                    </div>
                                ) : (
                                    availableColumns.map((col) => {
                                        const isChecked = selectedColumns.includes(col);
                                        return (
                                            <div key={col} onClick={() => handleToggleColumn(col)} style={styles.checkboxItem(isChecked)}>
                                                {isChecked ? <CheckSquare size={16} color="#2563eb" /> : <Square size={16} color="#94a3b8" />}
                                                <span style={{ fontWeight: isChecked ? '700' : '500', color: '#1e293b', wordBreak: 'break-all', flex: 1 }}>
                                                    {col}
                                                </span>
                                            </div>
                                        );
                                    })
                                )}
                            </div>
                        </div>

                        {/* 4. Méthode de Scaling */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>4. Méthode de Normalisation</label>
                            <div style={styles.selectBox}>
                                <select value={selectedMethod} onChange={(e) => setSelectedMethod(e.target.value)} style={styles.select}>
                                    {SCALING_METHODS.map((m) => (
                                        <option key={m.value} value={m.value}>{m.label}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* Paramètres Min-Max */}
                        {selectedMethod === 'min_max' && (
                            <div style={{ display: 'flex', gap: '12px', marginBottom: '18px' }}>
                                <div style={{ flex: 1 }}>
                                    <label style={styles.label}>Feature Range Min</label>
                                    <input type="number" step="0.1" value={minRange} onChange={(e) => setMinRange(e.target.value)} style={styles.input} />
                                </div>
                                <div style={{ flex: 1 }}>
                                    <label style={styles.label}>Feature Range Max</label>
                                    <input type="number" step="0.1" value={maxRange} onChange={(e) => setMaxRange(e.target.value)} style={styles.input} />
                                </div>
                            </div>
                        )}

                        <button onClick={handleExecute} disabled={loading} style={{ ...styles.actionBtn, backgroundColor: loading ? '#94a3b8' : '#2563eb' }}>
                            {loading ? <RefreshCw className="animate-spin" size={16} /> : <Play size={16} />}
                            {loading
                                ? 'Normalisation dans PostgreSQL...'
                                : `Normaliser Caractéristiques (${selectedColumns.length > 0 ? selectedColumns.length : 'Toutes'})`
                            }
                        </button>

                        {errorMsg && (
                            <div style={{ marginTop: '14px', padding: '12px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '14px', color: '#991b1b', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <AlertTriangle size={16} color="#dc2626" />
                                <span>{errorMsg}</span>
                            </div>
                        )}
                    </div>

                    {/* PANEL DROIT: RAPPORT DÉTAILLÉ & HISTORIQUE DES VERSIONS */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', minWidth: 0 }}>

                        {/* RAPPORT DE LA VERSION SÉLECTIONNÉE */}
                        {selectedVersion ? (
                            <div style={styles.resultCard}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <CheckCircle2 size={18} color="#1e40af" />
                                        <span style={{ fontWeight: '800', color: '#1e40af', fontSize: '14px' }}>
                                            Rapport de Normalisation (PostgreSQL Table)
                                        </span>
                                    </div>
                                    <button onClick={() => setSelectedVersion(null)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}>
                                        <X size={18} />
                                    </button>
                                </div>

                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
                                    <span style={{ backgroundColor: '#bfdbfe', color: '#1e40af', fontFamily: 'monospace', fontSize: '12px', padding: '4px 8px', borderRadius: '8px', fontWeight: 'bold' }}>
                                        {selectedVersion.version_id}
                                    </span>
                                    <span style={{ backgroundColor: '#e0f2fe', color: '#0369a1', fontFamily: 'monospace', fontSize: '12px', padding: '4px 8px', borderRadius: '8px', fontWeight: 'bold' }}>
                                        Attraction: {selectedVersion.id_attraction || 'ALL'}
                                    </span>
                                </div>

                                <div style={{ fontSize: '12px', color: '#334155', marginBottom: '14px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                    <div><strong>Méthode appliquée :</strong> {selectedVersion.method_label_fr || selectedVersion.method}</div>
                                    <div><strong>Version Parent :</strong> <code style={{ backgroundColor: '#ffffff', padding: '2px 6px', borderRadius: '6px' }}>{selectedVersion.parent_version_id}</code></div>
                                    {selectedVersion.file_path && (
                                        <div><strong>Table SQL générée :</strong> <code style={{ backgroundColor: '#ffffff', padding: '2px 6px', borderRadius: '6px' }}>{selectedVersion.file_path}</code></div>
                                    )}
                                    {selectedVersion.stats && selectedVersion.stats.rows_affected && (
                                        <div><strong>Lignes transformées :</strong> <code style={{ backgroundColor: '#ffffff', padding: '2px 6px', borderRadius: '6px' }}>{selectedVersion.stats.rows_affected}</code></div>
                                    )}
                                </div>

                                {/* TABLEAU DES COLONNES NORMALISED */}
                                <div style={{ overflowX: 'auto' }}>
                                    <table style={styles.table}>
                                        <thead>
                                            <tr>
                                                <th style={styles.th}>Colonne Transformée</th>
                                                <th style={{ ...styles.th, textAlign: 'center' }}>Statut</th>
                                                <th style={{ ...styles.th, textAlign: 'center' }}>Méthode</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {(() => {
                                                const stats = selectedVersion.stats || {};
                                                const targetCols = selectedVersion.target_columns || stats.columns_scaled || [];

                                                if (targetCols.length === 0) {
                                                    return (
                                                        <tr>
                                                            <td colSpan="3" style={{ ...styles.td, textAlign: 'center', color: '#94a3b8' }}>
                                                                Toutes les colonnes numériques ont été transformées.
                                                            </td>
                                                        </tr>
                                                    );
                                                }

                                                return targetCols.map((col) => (
                                                    <tr key={col}>
                                                        <td style={styles.td}>
                                                            <code style={{ fontWeight: '600' }}>{col}</code>
                                                        </td>
                                                        <td style={{ ...styles.td, textAlign: 'center', color: '#16a34a', fontWeight: '700' }}>
                                                            ✓ Transformée
                                                        </td>
                                                        <td style={{ ...styles.td, textAlign: 'center', color: '#2563eb', fontWeight: '800' }}>
                                                            {selectedVersion.method || 'standard'}
                                                        </td>
                                                    </tr>
                                                ));
                                            })()}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ) : (
                            <div style={{ ...styles.card, borderStyle: 'dashed', textAlign: 'center', color: '#64748b', fontSize: '13px', padding: '24px' }}>
                                💡 <i>Sélectionnez une version ci-dessous pour consulter son rapport PostgreSQL.</i>
                            </div>
                        )}

                        {/* DANH SÁCH VERSION ĐÃ LỌC THEO ATTRACTION */}
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
                                    Aucune version enregistrée pour {selectedAttraction === 'ALL' ? 'tous les sites' : `l'attraction ${selectedAttraction}`}.
                                </p>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                    {filteredVersions.map((v) => {
                                        const isSelected = selectedVersion && selectedVersion.version_id === v.version_id;
                                        const attrValue = v.id_attraction;
                                        return (
                                            <div key={v.version_id} onClick={() => setSelectedVersion(v)} style={styles.versionItem(isSelected)}>
                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                                                        <span style={{ fontFamily: 'monospace', fontWeight: '800', color: '#2563eb', fontSize: '13px' }}>
                                                            {v.version_id}
                                                        </span>
                                                        <span style={{ backgroundColor: '#f1f5f9', color: '#475569', fontSize: '11px', padding: '2px 6px', borderRadius: '6px', fontWeight: '600' }}>
                                                            {v.method_label_fr || v.method}
                                                        </span>
                                                        {attrValue && (
                                                            <span style={{ backgroundColor: '#e0f2fe', color: '#0369a1', fontSize: '10px', padding: '2px 6px', borderRadius: '6px', fontWeight: '700' }}>
                                                                {attrValue}
                                                            </span>
                                                        )}
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