import React, { useState, useEffect, useMemo } from 'react';
import {
    Sliders, Filter, Layers, CheckSquare, Square,
    Play, Trash2, AlertTriangle, CheckCircle2, X
} from 'lucide-react';

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

    // 1. Tự động đồng bộ Scope Attraction khi đổi Version Parent
    useEffect(() => {
        if (selectedParent && selectedParent !== 'v0_raw' && versions[selectedParent]) {
            const parentAttr = versions[selectedParent].id_attraction;
            if (parentAttr) {
                setSelectedAttraction(parentAttr);
            }
        }
    }, [selectedParent, versions]);

    // 2. Fetch stats dựa trên selectedParent và selectedAttraction (không reset bậy)
    useEffect(() => {
        const fetchParentStats = async () => {
            try {
                const res = await fetch(`${API_BASE}/versions/${selectedParent}/stats?id_attraction=${selectedAttraction}`);
                if (res.ok) {
                    const data = await res.json();
                    const nullCounts = data.null_counts || {};
                    setColumnsNullInfo(nullCounts);

                    // Chỉ cập nhật danh sách khả thi nếu backend thực sự trả về mảng hợp lệ
                    if (Array.isArray(data.available_attractions) && data.available_attractions.length > 0) {
                        setAvailableAttractions(data.available_attractions);
                    }

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

    // Lọc danh sách phiên bản hiển thị theo id_attraction đang chọn
    const filteredVersions = useMemo(() => {
        return Object.values(versions).filter(v => {
            if (selectedAttraction === 'ALL') return true;
            return String(v.id_attraction) === String(selectedAttraction);
        });
    }, [versions, selectedAttraction]);

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

        toggleBtn: (isActive) => ({
            flex: 1, padding: '10px 14px', borderRadius: '14px', fontSize: '12px', fontWeight: 'bold', cursor: 'pointer',
            transition: 'all 0.2s', border: '1px solid', borderColor: isActive ? '#2563eb' : '#e2e8f0',
            backgroundColor: isActive ? '#2563eb' : '#ffffff', color: isActive ? '#ffffff' : '#64748b',
        }),

        checkboxContainer: {
            maxHeight: '200px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '8px', backgroundColor: '#f8fafc', display: 'flex', flexDirection: 'column', gap: '6px'
        },
        checkboxItem: (isChecked) => ({
            display: 'flex', alignItems: 'center', padding: '8px 12px', borderRadius: '12px', cursor: 'pointer', fontSize: '13px',
            border: '1px solid', borderColor: isChecked ? '#93c5fd' : 'transparent', backgroundColor: isChecked ? '#eff6ff' : '#ffffff',
            transition: 'all 0.15s ease', gap: '8px'
        }),

        actionBtn: { width: '100%', padding: '12px 18px', backgroundColor: '#2563eb', color: '#ffffff', fontWeight: '700', border: 'none', borderRadius: '16px', cursor: 'pointer', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', transition: 'background-color 0.2s', marginTop: '8px' },
        smallBtn: { padding: '4px 8px', fontSize: '11px', fontWeight: 'bold', backgroundColor: '#ffffff', color: '#475569', border: '1px solid #e2e8f0', borderRadius: '8px', cursor: 'pointer' },

        resultCard: { backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '24px', padding: '20px', minWidth: 0 },
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
                                    Imputation & Traitement des Valeurs Manquantes
                                </h1>
                                <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', lineHeight: '1.2' }}>
                                    Nettoyage ciblé par colonne ou par règle métier à l'échelle d'une attraction ou du parc
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* GRID CONTENT */}
                <div style={styles.gridContainer}>

                    {/* PANEL CAU HINH */}
                    <div style={styles.card}>
                        <h2 style={{ margin: '0 0 16px 0', fontSize: '15px', fontWeight: '700', color: '#0f172a', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' }}>
                            ⚙️ Configuration du Traitement
                        </h2>

                        {/* 1. Parent Version */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>1. Version Source (Parent)</label>
                            <div style={styles.selectBox}>
                                <Layers size={14} color="#94a3b8" />
                                <select value={selectedParent} onChange={(e) => setSelectedParent(e.target.value)} style={styles.select}>
                                    <option value="v0_raw">v0_raw (Données brutes)</option>
                                    {Object.keys(versions).map((vId) => (
                                        <option key={vId} value={vId}>{vId}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* 2. Scope Attraction */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>2. Scope Attraction (id_attraction)</label>
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

                        {/* 3. Treatment Mode */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>3. Type d'approche</label>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <button type="button" onClick={() => setTreatmentMode('by_column')} style={styles.toggleBtn(treatmentMode === 'by_column')}>
                                    📊 Multi-Colonnes
                                </button>
                                <button type="button" onClick={() => setTreatmentMode('domain_rules')} style={styles.toggleBtn(treatmentMode === 'domain_rules')}>
                                    🌟 Règles Métier
                                </button>
                            </div>
                        </div>

                        {treatmentMode === 'by_column' && (
                            <>
                                {/* 4. Select Columns */}
                                <div style={styles.formGroup}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                        <label style={{ ...styles.label, margin: 0 }}>4. Colonnes ({selectedColumns.length})</label>
                                        <div style={{ display: 'flex', gap: '4px' }}>
                                            <button type="button" onClick={handleSelectAllColumns} style={styles.smallBtn}>Tout</button>
                                            <button type="button" onClick={handleDeselectAllColumns} style={styles.smallBtn}>Aucun</button>
                                        </div>
                                    </div>

                                    <div style={styles.checkboxContainer}>
                                        {availableColumns.map((col) => {
                                            const nullCount = columnsNullInfo[col];
                                            const isChecked = selectedColumns.includes(col);
                                            return (
                                                <div key={col} onClick={() => handleToggleColumn(col)} style={styles.checkboxItem(isChecked)}>
                                                    {isChecked ? <CheckSquare size={16} color="#2563eb" /> : <Square size={16} color="#94a3b8" />}
                                                    <span style={{ fontWeight: isChecked ? '700' : '500', color: '#1e293b', wordBreak: 'break-all', flex: 1 }}>
                                                        {col}
                                                    </span>
                                                    {nullCount !== undefined && (
                                                        <span style={{
                                                            fontSize: '11px', fontWeight: '800',
                                                            color: nullCount > 0 ? '#dc2626' : '#16a34a',
                                                            backgroundColor: nullCount > 0 ? '#fef2f2' : '#f0fdf4',
                                                            padding: '2px 8px', borderRadius: '12px'
                                                        }}>
                                                            {nullCount} nulls
                                                        </span>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* 5. Select Method */}
                                <div style={styles.formGroup}>
                                    <label style={styles.label}>5. Méthode d'imputation</label>
                                    <div style={styles.selectBox}>
                                        <select value={selectedMethod} onChange={(e) => setSelectedMethod(e.target.value)} style={styles.select}>
                                            {STAT_METHODS.map((m) => (
                                                <option key={m.value} value={m.value}>{m.label}</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>

                                {/* Dynamic Inputs */}
                                {selectedMethod === 'constant' && (
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>Valeur constante</label>
                                        <input type="number" value={constantVal} onChange={(e) => setConstantVal(Number(e.target.value))} style={styles.input} />
                                    </div>
                                )}
                                {selectedMethod === 'knn' && (
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>Nombre de voisins (KNN)</label>
                                        <input type="number" value={knnNeighbors} onChange={(e) => setKnnNeighbors(Number(e.target.value))} style={styles.input} />
                                    </div>
                                )}
                                {selectedMethod === 'polynomial' && (
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>Degré du polynôme</label>
                                        <input type="number" value={polyOrder} onChange={(e) => setPolyOrder(Number(e.target.value))} style={styles.input} />
                                    </div>
                                )}
                            </>
                        )}

                        {treatmentMode === 'domain_rules' && (
                            <div style={styles.formGroup}>
                                <label style={styles.label}>4. Règle métier globale</label>
                                <div style={styles.selectBox}>
                                    <select value={globalRule} onChange={(e) => setGlobalRule(e.target.value)} style={styles.select}>
                                        <option value="visitor_domain_rules">Règles fréquentation & météo</option>
                                        <option value="energy_domain_rules">Règles énergie & capteurs</option>
                                    </select>
                                </div>
                            </div>
                        )}

                        <button onClick={handleExecute} disabled={loading} style={{ ...styles.actionBtn, backgroundColor: loading ? '#94a3b8' : '#2563eb' }}>
                            <Play size={16} />
                            {loading ? 'Application en cours...' : `Lancer l'imputation (${selectedColumns.length})`}
                        </button>

                        {errorMsg && (
                            <div style={{ marginTop: '14px', padding: '12px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '14px', color: '#991b1b', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <AlertTriangle size={16} color="#dc2626" />
                                <span>{errorMsg}</span>
                            </div>
                        )}
                    </div>

                    {/* PANEL PHAI: KET QUA & DANH SACH VERSION */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', minWidth: 0 }}>

                        {/* CHI TIET VERSION DUOC CHON */}
                        {selectedVersion ? (
                            <div style={styles.resultCard}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <CheckCircle2 size={18} color="#166534" />
                                        <span style={{ fontWeight: '800', color: '#166534', fontSize: '14px' }}>Rapport Avant / Après</span>
                                    </div>
                                    <button onClick={() => setSelectedVersion(null)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}>
                                        <X size={18} />
                                    </button>
                                </div>

                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
                                    <span style={{ backgroundColor: '#bbf7d0', color: '#166534', fontFamily: 'monospace', fontSize: '12px', padding: '4px 8px', borderRadius: '8px', fontWeight: 'bold' }}>
                                        {selectedVersion.version_id}
                                    </span>
                                    <span style={{ backgroundColor: '#e0f2fe', color: '#0369a1', fontFamily: 'monospace', fontSize: '12px', padding: '4px 8px', borderRadius: '8px', fontWeight: 'bold' }}>
                                        Attraction: {selectedVersion.id_attraction || 'ALL'}
                                    </span>
                                </div>

                                <div style={{ fontSize: '12px', color: '#334155', marginBottom: '14px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                    <div><strong>Méthode :</strong> {selectedVersion.method_label_fr}</div>
                                    <div><strong>Parent :</strong> <code style={{ backgroundColor: '#ffffff', padding: '2px 6px', borderRadius: '6px' }}>{selectedVersion.parent_version_id}</code></div>
                                </div>

                                <div style={{ overflowX: 'auto' }}>
                                    <table style={styles.table}>
                                        <thead>
                                            <tr>
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
                                                    <tr key={col}>
                                                        <td style={styles.td}><code style={{ fontWeight: '600' }}>{col}</code></td>
                                                        <td style={{ ...styles.td, textAlign: 'center', color: '#dc2626', fontWeight: '800' }}>{before}</td>
                                                        <td style={{ ...styles.td, textAlign: 'center', color: '#16a34a', fontWeight: '800' }}>{after}</td>
                                                        <td style={{ ...styles.td, textAlign: 'center', color: '#2563eb', fontWeight: '800' }}>-{before - after}</td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ) : (
                            <div style={{ ...styles.card, borderStyle: 'dashed', textAlign: 'center', color: '#64748b', fontSize: '13px', padding: '24px' }}>
                                💡 <i>Sélectionnez une version ci-dessous pour consulter son rapport détaillé.</i>
                            </div>
                        )}

                        {/* DANH SACH VERSION DA LOC THEO ATTRACTION */}
                        <div style={styles.card}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' }}>
                                <h2 style={{ margin: 0, fontSize: '15px', fontWeight: '700', color: '#0f172a' }}>
                                    📂 Versions Enregistrées
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
                                        return (
                                            <div key={v.version_id} onClick={() => setSelectedVersion(v)} style={styles.versionItem(isSelected)}>
                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                                                        <span style={{ fontFamily: 'monospace', fontWeight: '800', color: '#2563eb', fontSize: '13px' }}>
                                                            {v.version_id}
                                                        </span>
                                                        <span style={{ backgroundColor: '#f1f5f9', color: '#475569', fontSize: '11px', padding: '2px 6px', borderRadius: '6px', fontWeight: '600' }}>
                                                            {v.method_label_fr}
                                                        </span>
                                                        {v.id_attraction && (
                                                            <span style={{ backgroundColor: '#e0f2fe', color: '#0369a1', fontSize: '10px', padding: '2px 6px', borderRadius: '6px', fontWeight: '700' }}>
                                                                {v.id_attraction}
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