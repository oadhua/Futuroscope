import React, { useState, useEffect } from 'react';

const STAT_METHODS = [
    { value: 'mean', label: 'Moyenne (Mean)' },
    { value: 'median', label: 'Médiane (Median)' },
    { value: 'mode', label: 'Mode' },
    { value: 'constant', label: 'Valeur Constante' },
    { value: 'bfill', label: 'Remplissage en arrière (bfill)' },
    { value: 'ffill', label: 'Remplissage en avant (ffill)' },
    { value: 'knn', label: 'K-Nearest Neighbors (KNN)' },
    { value: 'linear', label: 'Interpolation linéaire' },
    { value: 'polynomial', label: 'Interpolation polynomiale' },
];

export default function MissingValueModule() {
    const [versions, setVersions] = useState({});
    const [selectedParent, setSelectedParent] = useState('v0_raw');
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [availableAttractions, setAvailableAttractions] = useState([]);

    const [availableColumns, setAvailableColumns] = useState([]);
    const [columnsNullInfo, setColumnsNullInfo] = useState({});

    const [treatmentMode, setTreatmentMode] = useState('by_column');
    const [selectedColumns, setSelectedColumns] = useState([]);
    const [selectedMethod, setSelectedMethod] = useState('mean');
    const [globalRule, setGlobalRule] = useState('visitor_domain_rules');

    const [constantVal, setConstantVal] = useState(0);
    const [knnNeighbors, setKnnNeighbors] = useState(5);
    const [polyOrder, setPolyOrder] = useState(2);

    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);
    const [selectedVersion, setSelectedVersion] = useState(null);

    const API_BASE = 'http://localhost:8000/data-prep';

    const fetchVersions = async () => {
        try {
            const res = await fetch(`${API_BASE}/versions`);
            if (res.ok) {
                const data = await res.json();
                setVersions(data);
            }
        } catch (err) {
            console.error('Erreur lors du chargement des versions :', err);
        }
    };

    useEffect(() => {
        fetchVersions();
    }, []);

    useEffect(() => {
        const fetchParentStats = async () => {
            try {
                const res = await fetch(`${API_BASE}/versions/${selectedParent}/stats?id_attraction=${selectedAttraction}`);
                if (res.ok) {
                    const data = await res.json();
                    const nullCounts = data.null_counts || {};
                    setColumnsNullInfo(nullCounts);
                    setAvailableAttractions(data.available_attractions || []);

                    const cols = Object.keys(nullCounts);
                    setAvailableColumns(cols);

                    const defaultSelected = cols.filter(col => nullCounts[col] > 0);
                    setSelectedColumns(defaultSelected.length > 0 ? defaultSelected : (cols.length > 0 ? [cols[0]] : []));
                }
            } catch (err) {
                console.error('Erreur stats :', err);
            }
        };

        if (selectedParent) {
            fetchParentStats();
        }
    }, [selectedParent, selectedAttraction]);

    const handleToggleColumn = (col) => {
        if (selectedColumns.includes(col)) {
            setSelectedColumns(selectedColumns.filter((c) => c !== col));
        } else {
            setSelectedColumns([...selectedColumns, col]);
        }
    };

    const handleSelectAllColumns = () => setSelectedColumns([...availableColumns]);
    const handleDeselectAllColumns = () => setSelectedColumns([]);

    const handleExecute = async () => {
        if (treatmentMode === 'by_column' && selectedColumns.length === 0) {
            setErrorMsg('Veuillez sélectionner au moins une colonne à traiter.');
            return;
        }

        setLoading(true);
        setErrorMsg(null);

        let methodToSend = selectedMethod;
        let targetCols = selectedColumns;
        let params = {};

        if (treatmentMode === 'domain_rules') {
            methodToSend = globalRule;
            targetCols = [];
        } else {
            if (selectedMethod === 'constant') params['constant_value'] = constantVal;
            if (selectedMethod === 'knn') params['n_neighbors'] = knnNeighbors;
            if (selectedMethod === 'polynomial') params['order'] = polyOrder;
        }

        const payload = {
            parent_version_id: selectedParent,
            id_attraction: selectedAttraction,
            target_columns: targetCols,
            method: methodToSend,
            params: params,
        };

        try {
            const res = await fetch(`${API_BASE}/missing-values`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Erreur lors du traitement.');
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

    const handleDeleteVersion = async (e, versionId) => {
        e.stopPropagation();
        if (!window.confirm(`Supprimer la version ${versionId} ?`)) return;

        try {
            const res = await fetch(`${API_BASE}/versions/${versionId}`, { method: 'DELETE' });
            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Impossible de supprimer.');
            }
            if (selectedVersion && selectedVersion.version_id === versionId) {
                setSelectedVersion(null);
            }
            fetchVersions();
        } catch (err) {
            alert(err.message);
        }
    };

    return (
        <div style={styles.container}>
            <div style={styles.header}>
                <h1 style={styles.title}>⚙️ Data Prep : Imputation par Colonnes & Attraction</h1>
                <p style={styles.subtitle}>
                    Sélectionnez une attraction spécifique ou l'ensemble du parc (ALL) pour appliquer l'imputation sur plusieurs colonnes.
                </p>
            </div>

            <div style={styles.gridContainer}>
                {/* 1. CAU HINH */}
                <div style={{ ...styles.card, minWidth: 0 }}>
                    <h2 style={styles.cardTitle}>🎯 Configuration du Traitement</h2>

                    <div style={styles.formGroup}>
                        <label style={styles.label}>1. Version source (Parent)</label>
                        <select
                            value={selectedParent}
                            onChange={(e) => setSelectedParent(e.target.value)}
                            style={styles.input}
                        >
                            <option value="v0_raw">v0_raw (Données brutes)</option>
                            {Object.keys(versions).map((vId) => (
                                <option key={vId} value={vId}>{vId}</option>
                            ))}
                        </select>
                    </div>

                    <div style={styles.formGroup}>
                        <label style={styles.label}>2. Scope Attraction (id_attraction)</label>
                        <select
                            value={selectedAttraction}
                            onChange={(e) => setSelectedAttraction(e.target.value)}
                            style={{ ...styles.input, backgroundColor: '#f0f9ff', borderColor: '#0284c7' }}
                        >
                            <option value="ALL">🌐 Tous les sites (ALL)</option>
                            {availableAttractions.map((attr) => (
                                <option key={attr} value={attr}>🎢 Attraction ID: {attr}</option>
                            ))}
                        </select>
                    </div>

                    <div style={styles.formGroup}>
                        <label style={styles.label}>3. Type d'approche</label>
                        <div style={{ display: 'flex', gap: '10px' }}>
                            <button
                                type="button"
                                onClick={() => setTreatmentMode('by_column')}
                                style={{
                                    ...styles.toggleBtn,
                                    backgroundColor: treatmentMode === 'by_column' ? '#2563eb' : '#f1f5f9',
                                    color: treatmentMode === 'by_column' ? '#ffffff' : '#475569',
                                }}
                            >
                                📊 Sélection Multi-Colonnes
                            </button>
                            <button
                                type="button"
                                onClick={() => setTreatmentMode('domain_rules')}
                                style={{
                                    ...styles.toggleBtn,
                                    backgroundColor: treatmentMode === 'domain_rules' ? '#2563eb' : '#f1f5f9',
                                    color: treatmentMode === 'domain_rules' ? '#ffffff' : '#475569',
                                }}
                            >
                                🌟 Règles Métier Globale
                            </button>
                        </div>
                    </div>

                    {treatmentMode === 'by_column' && (
                        <>
                            <div style={styles.formGroup}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                                    <label style={styles.label}>
                                        4. Colonnes à traiter ({selectedColumns.length})
                                    </label>
                                    <div style={{ display: 'flex', gap: '6px' }}>
                                        <button type="button" onClick={handleSelectAllColumns} style={styles.smallActionBtn}>Tout</button>
                                        <button type="button" onClick={handleDeselectAllColumns} style={styles.smallActionBtn}>Aucun</button>
                                    </div>
                                </div>

                                <div style={styles.checkboxContainer}>
                                    {availableColumns.map((col) => {
                                        const nullCount = columnsNullInfo[col];
                                        const isChecked = selectedColumns.includes(col);
                                        return (
                                            <label
                                                key={col}
                                                style={{
                                                    ...styles.checkboxItem,
                                                    backgroundColor: isChecked ? '#eff6ff' : '#ffffff',
                                                    borderColor: isChecked ? '#3b82f6' : '#e2e8f0',
                                                }}
                                            >
                                                <input
                                                    type="checkbox"
                                                    checked={isChecked}
                                                    onChange={() => handleToggleColumn(col)}
                                                    style={{ marginRight: '8px', cursor: 'pointer', flexShrink: 0 }}
                                                />
                                                <span style={{ fontWeight: isChecked ? 'bold' : 'normal', color: '#1e293b', wordBreak: 'break-all' }}>
                                                    {col}
                                                </span>
                                                {nullCount !== undefined && (
                                                    <span style={{
                                                        marginLeft: 'auto',
                                                        fontSize: '11px',
                                                        fontWeight: 'bold',
                                                        color: nullCount > 0 ? '#dc2626' : '#16a34a',
                                                        backgroundColor: nullCount > 0 ? '#fef2f2' : '#f0fdf4',
                                                        padding: '2px 6px',
                                                        borderRadius: '4px',
                                                        flexShrink: 0
                                                    }}>
                                                        {nullCount} nulls
                                                    </span>
                                                )}
                                            </label>
                                        );
                                    })}
                                </div>
                            </div>

                            <div style={styles.formGroup}>
                                <label style={styles.label}>5. Méthode d'imputation</label>
                                <select
                                    value={selectedMethod}
                                    onChange={(e) => setSelectedMethod(e.target.value)}
                                    style={styles.input}
                                >
                                    {STAT_METHODS.map((m) => (
                                        <option key={m.value} value={m.value}>{m.label}</option>
                                    ))}
                                </select>
                            </div>

                            {selectedMethod === 'constant' && (
                                <div style={styles.formGroup}>
                                    <label style={styles.label}>Valeur constante</label>
                                    <input
                                        type="number"
                                        value={constantVal}
                                        onChange={(e) => setConstantVal(Number(e.target.value))}
                                        style={styles.input}
                                    />
                                </div>
                            )}

                            {selectedMethod === 'knn' && (
                                <div style={styles.formGroup}>
                                    <label style={styles.label}>Nombre de voisins (KNN)</label>
                                    <input
                                        type="number"
                                        value={knnNeighbors}
                                        onChange={(e) => setKnnNeighbors(Number(e.target.value))}
                                        style={styles.input}
                                    />
                                </div>
                            )}

                            {selectedMethod === 'polynomial' && (
                                <div style={styles.formGroup}>
                                    <label style={styles.label}>Degré du polynôme</label>
                                    <input
                                        type="number"
                                        value={polyOrder}
                                        onChange={(e) => setPolyOrder(Number(e.target.value))}
                                        style={styles.input}
                                    />
                                </div>
                            )}
                        </>
                    )}

                    {treatmentMode === 'domain_rules' && (
                        <div style={styles.formGroup}>
                            <label style={styles.label}>4. Choisir la règle métier</label>
                            <select
                                value={globalRule}
                                onChange={(e) => setGlobalRule(e.target.value)}
                                style={styles.input}
                            >
                                <option value="visitor_domain_rules">
                                    Règles Fréquentation & Météo
                                </option>
                                <option value="energy_domain_rules">
                                    Règles Énergie & Capteurs
                                </option>
                            </select>
                        </div>
                    )}

                    <button
                        onClick={handleExecute}
                        disabled={loading}
                        style={loading ? { ...styles.button, backgroundColor: '#94a3b8' } : styles.button}
                    >
                        {loading ? 'Application en cours...' : `▶ Traiter (${selectedColumns.length} colonne(s))`}
                    </button>

                    {errorMsg && <div style={styles.errorBox}>⚠️ {errorMsg}</div>}
                </div>

                {/* 2. HISTORIQUE & CHI TIET */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', minWidth: 0 }}>

                    {selectedVersion ? (
                        <div style={styles.resultCard}>
                            <div style={styles.resultHeader}>
                                <div>
                                    <span style={{ fontWeight: 'bold', color: '#166534', fontSize: '15px' }}>
                                        🔍 Détails de la version sélectionnée
                                    </span>
                                    <div style={{ marginTop: '4px', display: 'flex', gap: '6px' }}>
                                        <span style={styles.badge}>{selectedVersion.version_id}</span>
                                        <span style={styles.attrBadge}>Attraction: {selectedVersion.id_attraction || 'ALL'}</span>
                                    </div>
                                </div>
                                <button onClick={() => setSelectedVersion(null)} style={styles.closeBtn}>✕</button>
                            </div>

                            <div style={{ fontSize: '13px', color: '#334155', marginBottom: '12px' }}>
                                <p style={{ margin: '2px 0' }}><strong>Méthode :</strong> {selectedVersion.method_label_fr}</p>
                                <p style={{ margin: '2px 0' }}><strong>Parent :</strong> <code>{selectedVersion.parent_version_id}</code></p>
                                <p style={{ margin: '2px 0', wordBreak: 'break-all' }}>
                                    <strong>Colonnes traitées :</strong> <code>{selectedVersion.target_columns?.length > 0 ? selectedVersion.target_columns.join(', ') : 'Toutes les colonnes'}</code>
                                </p>
                                <p style={{ margin: '2px 0', wordBreak: 'break-all' }}>
                                    <strong>URI Database :</strong> <code style={styles.codePath}>{selectedVersion.file_path}</code>
                                </p>
                            </div>

                            <h4 style={styles.tableTitle}>📊 Statistique des Missing Values (Avant vs Après)</h4>
                            <div style={{ overflowX: 'auto' }}>
                                <table style={styles.table}>
                                    <thead>
                                        <tr style={{ backgroundColor: '#dcfce7' }}>
                                            <th style={styles.th}>Colonne</th>
                                            <th style={{ ...styles.th, textAlign: 'center' }}>Avant</th>
                                            <th style={{ ...styles.th, textAlign: 'center' }}>Après</th>
                                            <th style={{ ...styles.th, textAlign: 'center' }}>Nettoyé</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {selectedVersion.stats?.null_count_before && Object.keys(selectedVersion.stats.null_count_before).map((col) => {
                                            const before = selectedVersion.stats.null_count_before[col] || 0;
                                            const after = selectedVersion.stats.null_count_after ? (selectedVersion.stats.null_count_after[col] ?? 0) : 0;
                                            return (
                                                <tr key={col} style={{ borderBottom: '1px solid #e2e8f0' }}>
                                                    <td style={{ ...styles.td, wordBreak: 'break-all' }}><code>{col}</code></td>
                                                    <td style={{ ...styles.td, textAlign: 'center', color: '#dc2626', fontWeight: 'bold' }}>{before}</td>
                                                    <td style={{ ...styles.td, textAlign: 'center', color: '#16a34a', fontWeight: 'bold' }}>{after}</td>
                                                    <td style={{ ...styles.td, textAlign: 'center', color: '#2563eb', fontWeight: 'bold' }}>-{before - after}</td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    ) : (
                        <div style={styles.emptyPromptCard}>
                            💡 <i>Cliquez sur une version dans l'historique ci-dessous pour afficher le rapport Avant / Après traitement.</i>
                        </div>
                    )}

                    <div style={styles.card}>
                        <h2 style={styles.cardTitle}>📂 Historique des versions enregistrées</h2>

                        {Object.keys(versions).length === 0 ? (
                            <p style={{ color: '#94a3b8', fontStyle: 'italic', fontSize: '14px' }}>Aucune version enregistrée.</p>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                {Object.values(versions).map((v) => {
                                    const isSelected = selectedVersion && selectedVersion.version_id === v.version_id;
                                    return (
                                        <div
                                            key={v.version_id}
                                            onClick={() => setSelectedVersion(v)}
                                            style={{
                                                ...styles.versionItem,
                                                borderColor: isSelected ? '#2563eb' : '#e2e8f0',
                                                backgroundColor: isSelected ? '#eff6ff' : '#f8fafc',
                                            }}
                                        >
                                            <div style={{ flex: 1, minWidth: 0 }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                                    <span style={{ fontFamily: 'monospace', fontWeight: 'bold', color: '#2563eb' }}>{v.version_id}</span>
                                                    <span style={styles.smallBadge}>{v.method_label_fr}</span>
                                                    <span style={styles.attrBadge}>ID: {v.id_attraction || 'ALL'}</span>
                                                </div>
                                                <p style={{ fontSize: '12px', color: '#64748b', margin: '4px 0 0 0', wordBreak: 'break-all' }}>
                                                    Parent : <code>{v.parent_version_id}</code> | Colonnes : {v.target_columns?.length > 0 ? v.target_columns.join(', ') : 'Global'}
                                                </p>
                                            </div>

                                            <button onClick={(e) => handleDeleteVersion(e, v.version_id)} style={styles.deleteBtn}>🗑️</button>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>

                </div>
            </div>
        </div>
    );
}

const styles = {
    container: { padding: '24px', maxWidth: '1280px', margin: '0 auto', fontFamily: 'Segoe UI, sans-serif', backgroundColor: '#f8fafc', minHeight: '100vh', boxSizing: 'border-box' },
    header: { borderBottom: '1px solid #e2e8f0', paddingBottom: '16px', marginBottom: '24px' },
    title: { fontSize: '22px', fontWeight: 'bold', color: '#1e293b', margin: 0 },
    subtitle: { fontSize: '14px', color: '#64748b', marginTop: '4px' },
    gridContainer: { display: 'grid', gridTemplateColumns: 'minmax(320px, 1fr) minmax(400px, 1.6fr)', gap: '24px', alignItems: 'start' },
    card: { backgroundColor: '#ffffff', padding: '20px', borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', boxSizing: 'border-box' },
    cardTitle: { fontSize: '16px', fontWeight: 'bold', color: '#334155', marginBottom: '16px', borderBottom: '1px solid #f1f5f9', paddingBottom: '8px' },
    formGroup: { marginBottom: '16px' },
    label: { display: 'block', fontSize: '13px', fontWeight: '600', color: '#475569', marginBottom: '6px' },
    input: { width: '100%', padding: '8px 12px', backgroundColor: '#f8fafc', border: '1px solid #cbd5e1', borderRadius: '6px', fontSize: '14px', boxSizing: 'border-box' },
    toggleBtn: { flex: 1, padding: '8px', border: '1px solid #cbd5e1', borderRadius: '6px', fontWeight: 'bold', fontSize: '12px', cursor: 'pointer' },
    button: { width: '100%', padding: '10px 16px', backgroundColor: '#2563eb', color: '#ffffff', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '14px', marginTop: '8px' },
    errorBox: { marginTop: '12px', padding: '10px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '6px', color: '#991b1b', fontSize: '13px', wordBreak: 'break-all' },
    resultCard: { backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', padding: '16px', borderRadius: '12px', minWidth: 0 },
    emptyPromptCard: { padding: '16px', backgroundColor: '#ffffff', border: '1px dashed #cbd5e1', borderRadius: '12px', color: '#64748b', fontSize: '13px', textAlign: 'center' },
    resultHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' },
    closeBtn: { background: 'none', border: 'none', color: '#64748b', fontSize: '16px', cursor: 'pointer' },
    badge: { backgroundColor: '#bbf7d0', color: '#166534', fontFamily: 'monospace', fontSize: '12px', padding: '2px 8px', borderRadius: '4px' },
    attrBadge: { backgroundColor: '#e0f2fe', color: '#0369a1', fontFamily: 'monospace', fontSize: '11px', padding: '2px 6px', borderRadius: '4px', fontWeight: 'bold' },
    smallBadge: { backgroundColor: '#e2e8f0', color: '#475569', fontSize: '11px', padding: '2px 6px', borderRadius: '4px' },
    codePath: { backgroundColor: '#ffffff', padding: '2px 6px', borderRadius: '4px', border: '1px solid #e2e8f0', fontSize: '12px', wordBreak: 'break-all' },
    tableTitle: { fontSize: '13px', fontWeight: 'bold', color: '#166534', margin: '12px 0 6px 0' },
    table: { width: '100%', borderCollapse: 'collapse', backgroundColor: '#ffffff', borderRadius: '6px', overflow: 'hidden', fontSize: '13px' },
    th: { padding: '8px 12px', textAlign: 'left', color: '#334155' },
    td: { padding: '8px 12px' },
    versionItem: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', borderWidth: '1px', borderStyle: 'solid', borderRadius: '8px', cursor: 'pointer', gap: '12px' },
    deleteBtn: { backgroundColor: 'transparent', color: '#dc2626', border: 'none', cursor: 'pointer', fontSize: '16px', flexShrink: 0 },

    checkboxContainer: {
        maxHeight: '180px',
        overflowY: 'auto',
        overflowX: 'hidden',
        border: '1px solid #cbd5e1',
        borderRadius: '6px',
        padding: '6px',
        backgroundColor: '#f8fafc',
        display: 'flex',
        flexDirection: 'column',
        gap: '4px'
    },
    checkboxItem: {
        display: 'flex',
        alignItems: 'center',
        padding: '6px 10px',
        borderRadius: '4px',
        borderWidth: '1px',
        borderStyle: 'solid',
        cursor: 'pointer',
        fontSize: '13px',
        userSelect: 'none',
        gap: '4px'
    },
    smallActionBtn: {
        padding: '2px 8px',
        fontSize: '11px',
        fontWeight: 'bold',
        backgroundColor: '#e2e8f0',
        color: '#334155',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer'
    }
};