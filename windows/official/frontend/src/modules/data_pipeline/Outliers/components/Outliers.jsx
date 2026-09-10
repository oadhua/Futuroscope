import React, { useState, useEffect, useMemo } from 'react';
import {
    Sliders, Filter, Layers, CheckSquare, Square,
    Play, Trash2, AlertTriangle, CheckCircle2, X,
    ShieldAlert, RefreshCw, Search, Sparkles, BarChart2
} from 'lucide-react';

const STEP_TYPE = 'outlier_treatment';

const OUTLIER_METHODS = [
    { value: 'iqr', label: 'IQR - Écart Interquartile', desc: 'Détection basée sur les quartiles (Q1, Q3)' },
    { value: 'z_score', label: 'Z-Score (Écart-type)', desc: 'Détection basée sur l\'écart à la moyenne' },
    { value: 'isolation_forest', label: 'Isolation Forest', desc: 'Algorithme ML pour anomalies complexes' },
    { value: 'lof', label: 'LOF - Local Outlier Factor', desc: 'Densité locale par rapport aux voisins' },
];

export default function OutliersModule() {
    // 1. Quản lý danh sách Scope & Versions
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [selectedParent, setSelectedParent] = useState('v0_raw');
    const [allVersions, setAllVersions] = useState({});
    const [availableAttractions, setAvailableAttractions] = useState([]);

    // 2. Quản lý danh sách Cột & Thống kê Outliers
    const [availableColumns, setAvailableColumns] = useState([]);
    const [columnsOutliersInfo, setColumnsOutliersInfo] = useState({});
    const [columnSearch, setColumnSearch] = useState('');

    // 3. Cấu hình xử lý Outliers
    const [treatmentMode, setTreatmentMode] = useState('by_column'); // 'by_column' | 'domain_rules'
    const [selectedColumns, setSelectedColumns] = useState([]);
    const [selectedMethod, setSelectedMethod] = useState('iqr');
    const [selectedAction, setSelectedAction] = useState('cap');
    const [globalRule, setGlobalRule] = useState('visitor_domain_rules');

    // Hyperparameters
    const [zThreshold, setZThreshold] = useState(3.0);
    const [iqrFactor, setIqrFactor] = useState(1.5);
    const [contamination, setContamination] = useState(0.05);

    // Status & Output States
    const [loading, setLoading] = useState(false);
    const [fetchingStats, setFetchingStats] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);
    const [selectedVersion, setSelectedVersion] = useState(null);

    const API_BASE = 'http://localhost:8000/data-prep';

    // Fetch TẤT CẢ các versions từ backend
    const fetchVersions = async () => {
        try {
            const res = await fetch(`${API_BASE}/versions`);
            if (res.ok) {
                const data = await res.json();
                setAllVersions(data || {});
            }
        } catch (err) {
            console.error('Erreur lors du chargement des versions:', err);
        }
    };

    useEffect(() => {
        fetchVersions();
    }, []);

    // Lọc danh sách Version hiển thị ở CỘT BÊN PHẢI (Chỉ hiển thị các version thuộc Outliers)
    const moduleVersions = useMemo(() => {
        return Object.values(allVersions).filter(v => {
            const matchStep = v.step_type === STEP_TYPE || v.step_type === 'outlier_detection' || v.step_type === 'outliers';
            if (selectedAttraction === 'ALL') return matchStep;
            const attrVal = v.id_attraction || v.attraction_id;
            return matchStep && String(attrVal) === String(selectedAttraction);
        });
    }, [allVersions, selectedAttraction]);

    // Lọc danh sách Parent Version cho Dropdown
    const parentVersionsList = useMemo(() => {
        return Object.entries(allVersions).filter(([vId, vObj]) => {
            if (vId === 'v0_raw') return true;
            if (selectedAttraction === 'ALL') return true;

            const attrVal = vObj?.id_attraction || vObj?.attraction_id;
            return !attrVal || attrVal === 'ALL' || String(attrVal) === String(selectedAttraction);
        });
    }, [allVersions, selectedAttraction]);

    // Fetch Statistiques des Outliers từ Parent Version
    useEffect(() => {
        const fetchParentStats = async () => {
            setFetchingStats(true);
            try {
                const queryParams = new URLSearchParams({
                    id_attraction: selectedAttraction,
                    method: selectedMethod,
                    iqr_factor: iqrFactor,
                    z_threshold: zThreshold
                });

                const url = `${API_BASE}/outliers/versions/${selectedParent}/stats?${queryParams.toString()}`;
                const res = await fetch(url);

                if (res.ok) {
                    const data = await res.json();
                    const rawCounts = data.outlier_counts || data.stats?.outliers_detected || {};
                    let normalizedCounts = {};

                    if (Array.isArray(rawCounts)) {
                        rawCounts.forEach(item => {
                            if (typeof item === 'object' && item !== null) {
                                const colName = item.column || item.col || Object.keys(item)[0];
                                const cnt = item.count || item.outliers || item.n_outliers || Object.values(item)[0] || 0;
                                if (colName) normalizedCounts[colName] = cnt;
                            }
                        });
                    } else if (typeof rawCounts === 'object' && rawCounts !== null) {
                        normalizedCounts = rawCounts;
                    }

                    setColumnsOutliersInfo(normalizedCounts);

                    // Cập nhật danh sách attractions khả dụng khi đang chọn ALL
                    const attrs = data.available_attractions || [];
                    if (selectedAttraction === 'ALL' && Array.isArray(attrs) && attrs.length > 0) {
                        setAvailableAttractions(attrs);
                    }

                    const cols = Object.keys(normalizedCounts);
                    setAvailableColumns(cols);

                    // Mặc định chọn tất cả các cột nếu chưa chọn
                    if (cols.length > 0) {
                        setSelectedColumns(prev => prev.length === 0 ? cols : prev);
                    }
                }
            } catch (err) {
                console.error('Erreur lors du chargement des statistiques:', err);
            } finally {
                setFetchingStats(false);
            }
        };

        if (selectedParent) {
            fetchParentStats();
        }
    }, [selectedParent, selectedAttraction, selectedMethod, iqrFactor, zThreshold]);

    // Filtered Columns cho thanh tìm kiếm cột
    const filteredColumns = useMemo(() => {
        return availableColumns.filter(col =>
            col.toLowerCase().includes(columnSearch.toLowerCase())
        );
    }, [availableColumns, columnSearch]);

    // Handlers chọn cột
    const handleToggleColumn = (col) => {
        setSelectedColumns(prev =>
            prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
        );
    };

    const handleSelectAllColumns = () => setSelectedColumns([...availableColumns]);
    const handleDeselectAllColumns = () => setSelectedColumns([]);
    const handleSelectOutlierColumnsOnly = () => {
        const outlierCols = availableColumns.filter(col => (columnsOutliersInfo[col] || 0) > 0);
        setSelectedColumns(outlierCols);
    };

    // Thực thi xử lý Outliers
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
            if (selectedMethod === 'z_score') params['z_threshold'] = zThreshold;
            if (selectedMethod === 'iqr') params['iqr_factor'] = iqrFactor;
            if (selectedMethod === 'isolation_forest') params['contamination'] = contamination;
        }

        const payload = {
            parent_version_id: selectedParent,
            id_attraction: selectedAttraction,
            target_columns: targetCols,
            method: methodToSend,
            action: selectedAction,
            params: params,
        };

        try {
            const res = await fetch(`${API_BASE}/outliers`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Une erreur est survenue lors du traitement des outliers.');
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

    // Xóa Version
    const handleDeleteVersion = async (e, versionId) => {
        e.stopPropagation();
        if (!window.confirm(`Êtes-vous sûr de vouloir supprimer la version ${versionId} ?`)) return;

        try {
            const res = await fetch(`${API_BASE}/versions/${versionId}`, { method: 'DELETE' });
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

        toggleBtn: (isActive) => ({
            flex: 1, padding: '10px 14px', borderRadius: '14px', fontSize: '12px', fontWeight: 'bold', cursor: 'pointer',
            transition: 'all 0.2s', border: '1px solid', borderColor: isActive ? '#dc2626' : '#e2e8f0',
            backgroundColor: isActive ? '#dc2626' : '#ffffff', color: isActive ? '#ffffff' : '#64748b',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'
        }),

        checkboxContainer: {
            maxHeight: '220px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '8px', backgroundColor: '#f8fafc', display: 'flex', flexDirection: 'column', gap: '6px'
        },
        checkboxItem: (isChecked) => ({
            display: 'flex', alignItems: 'center', padding: '8px 12px', borderRadius: '12px', cursor: 'pointer', fontSize: '13px',
            border: '1px solid', borderColor: isChecked ? '#fca5a5' : 'transparent', backgroundColor: isChecked ? '#fef2f2' : '#ffffff',
            transition: 'all 0.15s ease', gap: '8px'
        }),

        actionBtn: { width: '100%', padding: '12px 18px', backgroundColor: '#dc2626', color: '#ffffff', fontWeight: '700', border: 'none', borderRadius: '16px', cursor: 'pointer', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', transition: 'background-color 0.2s', marginTop: '8px' },
        smallBtn: (active) => ({
            padding: '4px 8px', fontSize: '11px', fontWeight: 'bold',
            backgroundColor: active ? '#fee2e2' : '#ffffff',
            color: active ? '#991b1b' : '#475569',
            border: '1px solid', borderColor: active ? '#fca5a5' : '#e2e8f0',
            borderRadius: '8px', cursor: 'pointer', transition: 'all 0.15s ease'
        }),

        resultCard: { backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '24px', padding: '20px', minWidth: 0 },
        table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0, backgroundColor: '#ffffff', borderRadius: '16px', overflow: 'hidden', fontSize: '13px', border: '1px solid #e2e8f0' },
        th: { padding: '10px 14px', textAlign: 'left', color: '#475569', fontWeight: '700', borderBottom: '1px solid #e2e8f0', backgroundColor: '#f8fafc' },
        td: { padding: '10px 14px', borderBottom: '1px solid #f1f5f9' },

        versionItem: (isSelected) => ({
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px', borderRadius: '16px', cursor: 'pointer',
            border: '1px solid', borderColor: isSelected ? '#dc2626' : '#e2e8f0', backgroundColor: isSelected ? '#fef2f2' : '#ffffff',
            transition: 'all 0.15s ease', gap: '12px'
        })
    };

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* HEADER CARD */}
                <div style={styles.headerCard}>
                    <div style={styles.topBar}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={styles.iconBg('#fef2f2', '#dc2626')}>
                                <ShieldAlert size={24} />
                            </div>
                            <div>
                                <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2' }}>
                                    Détection & Traitement des Valeurs Aberrantes (Outliers)
                                </h1>
                                <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', lineHeight: '1.2' }}>
                                    Identification et filtrage statistique ou par règles métier des anomalies de fréquentation / capteurs
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* GRID CONTENT */}
                <div style={styles.gridContainer}>

                    {/* PANEL TRÁI: CẤU HÌNH XỬ LÝ */}
                    <div style={styles.card}>
                        <h2 style={{ margin: '0 0 16px 0', fontSize: '15px', fontWeight: '700', color: '#0f172a', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Sliders size={18} color="#dc2626" /> Configuration des Outliers
                        </h2>

                        {/* 1. Scope Attraction (ĐƯA LÊN ĐẦU) */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>1. Scope Attraction (id_attraction)</label>
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

                        {/* 2. Version Source (Parent) (ĐƯA XUỐNG DƯỚI) */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>2. Version Source (Parent)</label>
                            <div style={styles.selectBox}>
                                <Layers size={14} color="#94a3b8" />
                                <select value={selectedParent} onChange={(e) => setSelectedParent(e.target.value)} style={styles.select}>
                                    <option value="v0_raw">v0_raw</option>
                                    {parentVersionsList.map(([vId, vObj]) => {
                                        if (vId === 'v0_raw') return null;
                                        return (
                                            <option key={vId} value={vId}>
                                                {vId} {vObj?.step_type ? `(${vObj.step_type})` : ''}
                                            </option>
                                        );
                                    })}
                                </select>
                            </div>
                        </div>

                        {/* 3. Type d'approche */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>3. Type d'approche</label>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <button type="button" onClick={() => { setTreatmentMode('by_column'); setErrorMsg(null); }} style={styles.toggleBtn(treatmentMode === 'by_column')}>
                                    <BarChart2 size={14} /> Multi-Colonnes
                                </button>
                                <button type="button" onClick={() => { setTreatmentMode('domain_rules'); setErrorMsg(null); }} style={styles.toggleBtn(treatmentMode === 'domain_rules')}>
                                    <Sparkles size={14} /> Règles Métier
                                </button>
                            </div>
                        </div>

                        {treatmentMode === 'by_column' && (
                            <>
                                {/* 4. Sélection des Colonnes */}
                                <div style={styles.formGroup}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                        <label style={{ ...styles.label, margin: 0 }}>
                                            4. Colonnes ({selectedColumns.length}/{availableColumns.length})
                                        </label>
                                        <div style={{ display: 'flex', gap: '4px' }}>
                                            <button type="button" onClick={handleSelectOutlierColumnsOnly} style={styles.smallBtn(true)} title="Sélectionner uniquement les colonnes contenant des outliers">
                                                Avec aberrants
                                            </button>
                                            <button type="button" onClick={handleSelectAllColumns} style={styles.smallBtn(false)}>Tout</button>
                                            <button type="button" onClick={handleDeselectAllColumns} style={styles.smallBtn(false)}>Aucun</button>
                                        </div>
                                    </div>

                                    {/* Column Search Box */}
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '6px 10px', borderRadius: '12px', marginBottom: '8px' }}>
                                        <Search size={14} color="#94a3b8" />
                                        <input
                                            type="text"
                                            placeholder="Rechercher une colonne..."
                                            value={columnSearch}
                                            onChange={(e) => setColumnSearch(e.target.value)}
                                            style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '12px', width: '100%', color: '#0f172a' }}
                                        />
                                    </div>

                                    <div style={styles.checkboxContainer}>
                                        {fetchingStats ? (
                                            <div style={{ fontSize: '12px', color: '#94a3b8', padding: '12px', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                                                <RefreshCw className="animate-spin" size={14} /> Chargement des statistiques...
                                            </div>
                                        ) : filteredColumns.length === 0 ? (
                                            <div style={{ fontSize: '12px', color: '#94a3b8', padding: '8px', textAlign: 'center' }}>
                                                Aucune colonne disponible
                                            </div>
                                        ) : (
                                            filteredColumns.map((col) => {
                                                const outlierCount = columnsOutliersInfo[col] ?? 0;
                                                const isChecked = selectedColumns.includes(col);
                                                return (
                                                    <div key={col} onClick={() => handleToggleColumn(col)} style={styles.checkboxItem(isChecked)}>
                                                        {isChecked ? <CheckSquare size={16} color="#dc2626" /> : <Square size={16} color="#94a3b8" />}
                                                        <span style={{ fontWeight: isChecked ? '700' : '500', color: '#1e293b', wordBreak: 'break-all', flex: 1 }}>
                                                            {col}
                                                        </span>
                                                        <span style={{
                                                            fontSize: '11px', fontWeight: '800',
                                                            color: outlierCount > 0 ? '#dc2626' : '#16a34a',
                                                            backgroundColor: outlierCount > 0 ? '#fef2f2' : '#f0fdf4',
                                                            padding: '2px 8px', borderRadius: '12px'
                                                        }}>
                                                            {outlierCount} aberrants
                                                        </span>
                                                    </div>
                                                );
                                            })
                                        )}
                                    </div>
                                </div>

                                {/* 5. Méthode de Détection */}
                                <div style={styles.formGroup}>
                                    <label style={styles.label}>5. Méthode de Détection</label>
                                    <div style={styles.selectBox}>
                                        <select value={selectedMethod} onChange={(e) => setSelectedMethod(e.target.value)} style={styles.select}>
                                            {OUTLIER_METHODS.map((m) => (
                                                <option key={m.value} value={m.value}>{m.label}</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>

                                {/* 6. Action de Traitement */}
                                <div style={styles.formGroup}>
                                    <label style={styles.label}>6. Action de Traitement</label>
                                    <div style={styles.selectBox}>
                                        <select value={selectedAction} onChange={(e) => setSelectedAction(e.target.value)} style={styles.select}>
                                            <option value="cap">Cap / Winsorization (Écretage)</option>
                                            <option value="nullify">Nullify (Convertir en NaN pour Imputation)</option>
                                            <option value="drop">Drop (Supprimer les lignes contenant des outliers)</option>
                                            <option value="none">None (Inspecter uniquement)</option>
                                        </select>
                                    </div>
                                </div>

                                {/* Hyperparamètres */}
                                {selectedMethod === 'z_score' && (
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>Seuil Z-Score (Threshold)</label>
                                        <input type="number" step="0.1" value={zThreshold} onChange={(e) => setZThreshold(Number(e.target.value))} style={styles.input} />
                                    </div>
                                )}
                                {selectedMethod === 'iqr' && (
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>Facteur IQR (ex: 1.5 ou 3.0)</label>
                                        <input type="number" step="0.1" value={iqrFactor} onChange={(e) => setIqrFactor(Number(e.target.value))} style={styles.input} />
                                    </div>
                                )}
                                {selectedMethod === 'isolation_forest' && (
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>Taux de contamination (0.01 à 0.2)</label>
                                        <input type="number" step="0.01" value={contamination} onChange={(e) => setContamination(Number(e.target.value))} style={styles.input} />
                                    </div>
                                )}
                            </>
                        )}

                        {treatmentMode === 'domain_rules' && (
                            <div style={styles.formGroup}>
                                <label style={styles.label}>4. Règle métier globale</label>
                                <div style={styles.selectBox}>
                                    <select value={globalRule} onChange={(e) => setGlobalRule(e.target.value)} style={styles.select}>
                                        <option value="visitor_domain_rules">Règles fréquentation</option>
                                        <option value="energy_domain_rules">Règles énergie</option>
                                    </select>
                                </div>
                            </div>
                        )}

                        <button onClick={handleExecute} disabled={loading} style={{ ...styles.actionBtn, backgroundColor: loading ? '#94a3b8' : '#dc2626' }}>
                            {loading ? <RefreshCw className="animate-spin" size={16} /> : <Play size={16} />}
                            {loading
                                ? 'Traitement en cours...'
                                : treatmentMode === 'domain_rules'
                                    ? 'Traiter les Outliers (Règles Métier)'
                                    : `Traiter les Outliers (${selectedColumns.length})`
                            }
                        </button>

                        {errorMsg && (
                            <div style={{ marginTop: '14px', padding: '12px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '14px', color: '#991b1b', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <AlertTriangle size={16} color="#dc2626" />
                                <span>{errorMsg}</span>
                            </div>
                        )}
                    </div>

                    {/* PANEL PHẢI: BÁO CÁO CHI TIẾT & DANH SÁCH VERSION */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', minWidth: 0 }}>

                        {/* RAPPORT DE LA VERSION SÉLECTIONNÉE */}
                        {selectedVersion ? (
                            <div style={styles.resultCard}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <CheckCircle2 size={18} color="#991b1b" />
                                        <span style={{ fontWeight: '800', color: '#991b1b', fontSize: '14px' }}>
                                            Rapport de Traitement des Outliers
                                        </span>
                                    </div>
                                    <button onClick={() => setSelectedVersion(null)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}>
                                        <X size={18} />
                                    </button>
                                </div>

                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
                                    <span style={{ backgroundColor: '#fecaca', color: '#991b1b', fontFamily: 'monospace', fontSize: '12px', padding: '4px 8px', borderRadius: '8px', fontWeight: 'bold' }}>
                                        {selectedVersion.version_id}
                                    </span>
                                    <span style={{ backgroundColor: '#e0f2fe', color: '#0369a1', fontFamily: 'monospace', fontSize: '12px', padding: '4px 8px', borderRadius: '8px', fontWeight: 'bold' }}>
                                        Attraction: {selectedVersion.id_attraction || 'ALL'}
                                    </span>
                                </div>

                                <div style={{ fontSize: '12px', color: '#334155', marginBottom: '14px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                    <div><strong>Méthode :</strong> {selectedVersion.method_label_fr || selectedVersion.method}</div>
                                    <div><strong>Action effectuée :</strong> <code style={{ backgroundColor: '#ffffff', padding: '2px 6px', borderRadius: '6px' }}>{selectedVersion.action || 'cap'}</code></div>
                                    <div><strong>Version Parent :</strong> <code style={{ backgroundColor: '#ffffff', padding: '2px 6px', borderRadius: '6px' }}>{selectedVersion.parent_version_id}</code></div>
                                    {selectedVersion.parameters && Object.keys(selectedVersion.parameters).length > 0 && (
                                        <div><strong>Paramètres :</strong> <code style={{ backgroundColor: '#ffffff', padding: '2px 6px', borderRadius: '6px' }}>{JSON.stringify(selectedVersion.parameters)}</code></div>
                                    )}
                                </div>

                                {/* TABLEAU AVANT / APRÈS / NETTOYÉS */}
                                <div style={{ overflowX: 'auto' }}>
                                    <table style={styles.table}>
                                        <thead>
                                            <tr>
                                                <th style={styles.th}>Colonne</th>
                                                <th style={{ ...styles.th, textAlign: 'center' }}>Avant</th>
                                                <th style={{ ...styles.th, textAlign: 'center' }}>Après</th>
                                                <th style={{ ...styles.th, textAlign: 'center' }}>Traités</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {(() => {
                                                const stats = selectedVersion.stats || {};
                                                const beforeObj = stats.outliers_before || stats.outliers_detected || {};
                                                const afterObj = stats.outliers_after || {};

                                                const cols = Array.from(new Set([...Object.keys(beforeObj), ...Object.keys(afterObj)]));

                                                if (cols.length === 0) {
                                                    return (
                                                        <tr>
                                                            <td colSpan="4" style={{ ...styles.td, textAlign: 'center', color: '#94a3b8' }}>
                                                                Aucun détail statistique disponible.
                                                            </td>
                                                        </tr>
                                                    );
                                                }

                                                return cols.map((col) => {
                                                    const avant = beforeObj[col] ?? 0;
                                                    const apres = afterObj[col] ?? (selectedVersion.action === 'none' ? avant : 0);
                                                    const nettoyes = Math.max(0, avant - apres);

                                                    return (
                                                        <tr key={col}>
                                                            <td style={styles.td}>
                                                                <code style={{ fontWeight: '600' }}>{col}</code>
                                                            </td>
                                                            <td style={{ ...styles.td, textAlign: 'center', color: avant > 0 ? '#dc2626' : '#64748b', fontWeight: '700' }}>
                                                                {avant}
                                                            </td>
                                                            <td style={{ ...styles.td, textAlign: 'center', color: apres > 0 ? '#d97706' : '#16a34a', fontWeight: '700' }}>
                                                                {apres}
                                                            </td>
                                                            <td style={{ ...styles.td, textAlign: 'center', color: nettoyes > 0 ? '#2563eb' : '#64748b', fontWeight: '800' }}>
                                                                {nettoyes > 0 ? `-${nettoyes}` : '0'}
                                                            </td>
                                                        </tr>
                                                    );
                                                });
                                            })()}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ) : (
                            <div style={{ ...styles.card, borderStyle: 'dashed', textAlign: 'center', color: '#64748b', fontSize: '13px', padding: '24px' }}>
                                💡 <i>Sélectionnez une version ci-dessous pour consulter son rapport détaillé.</i>
                            </div>
                        )}

                        {/* DANH SÁCH VERSION ĐÃ LỌC THEO MODULE VÀ ATTRACTION */}
                        <div style={styles.card}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' }}>
                                <h2 style={{ margin: 0, fontSize: '15px', fontWeight: '700', color: '#0f172a' }}>
                                    📂 Versions Enregistrées
                                </h2>
                                <span style={{ fontSize: '11px', fontWeight: '700', color: '#0369a1', backgroundColor: '#e0f2fe', padding: '2px 8px', borderRadius: '10px' }}>
                                    {selectedAttraction === 'ALL' ? 'Tous les sites' : `Attraction: ${selectedAttraction}`}
                                </span>
                            </div>

                            {moduleVersions.length === 0 ? (
                                <p style={{ color: '#94a3b8', fontStyle: 'italic', fontSize: '13px', margin: 0 }}>
                                    Aucune version enregistrée pour {selectedAttraction === 'ALL' ? 'tous les sites' : `l'attraction ${selectedAttraction}`}.
                                </p>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                    {moduleVersions.map((v) => {
                                        const isSelected = selectedVersion && selectedVersion.version_id === v.version_id;
                                        const attrValue = v.id_attraction || v.attraction_id;
                                        return (
                                            <div key={v.version_id} onClick={() => setSelectedVersion(v)} style={styles.versionItem(isSelected)}>
                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                                                        <span style={{ fontFamily: 'monospace', fontWeight: '800', color: '#dc2626', fontSize: '13px' }}>
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