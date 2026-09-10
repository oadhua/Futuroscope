import React, { useState, useEffect, useMemo } from 'react';
import {
    CopyX, Filter, Layers, CheckSquare, Square,
    Play, Trash2, AlertTriangle, CheckCircle2, X, RefreshCw,
    BarChart2, Table, Activity, Search
} from 'lucide-react';

const STEP_TYPE = 'deduplication';

const KEEP_OPTIONS = [
    { value: 'first', label: 'Conserver la première occurrence (Recommandé)' },
    { value: 'last', label: 'Conserver la dernière occurrence' },
    { value: 'none', label: 'Supprimer toutes les occurrences dupliquées' },
];

export default function DeduplicationModule() {
    // 1. Quản lý danh sách Scope & Versions
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [selectedParent, setSelectedParent] = useState('v0_raw');
    const [allVersions, setAllVersions] = useState({});
    const [availableAttractions, setAvailableAttractions] = useState([]);

    // 2. Quản lý danh sách Cột & Thanh tìm kiếm
    const [availableColumns, setAvailableColumns] = useState([]);
    const [selectedColumns, setSelectedColumns] = useState([]);
    const [columnSearch, setColumnSearch] = useState('');

    // 3. Paramètres Deduplication & Stats
    const [keepOption, setKeepOption] = useState('first');
    const [totalDuplicates, setTotalDuplicates] = useState(0);
    const [totalRecords, setTotalRecords] = useState(0);

    // 4. Các trạng thái giao diện UI
    const [loading, setLoading] = useState(false);
    const [fetchingStats, setFetchingStats] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);
    const [selectedVersion, setSelectedVersion] = useState(null);
    const [activeTab, setActiveTab] = useState('summary');

    const API_BASE = 'http://localhost:8000/data-prep';

    // Fetch TẤT CẢ các versions từ backend Registry
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

    // Lọc danh sách Version hiển thị ở CỘT BÊN PHẢI (Chỉ giữ phiên bản thuộc Deduplication)
    const moduleVersions = useMemo(() => {
        return Object.values(allVersions).filter(v => {
            if (!v) return false;
            const step = (v.step_type || '').toLowerCase();
            const method = (v.method || '').toLowerCase();

            const matchStep = step === STEP_TYPE ||
                step === 'dedup' ||
                step === 'remove_duplicates' ||
                step === 'deduplication_module' ||
                method.includes('dedup') ||
                method.includes('duplicate');

            if (!matchStep) return false;
            if (selectedAttraction === 'ALL') return true;

            const attrVal = v.id_attraction !== undefined ? v.id_attraction : v.attraction_id;
            return !attrVal || attrVal === 'ALL' || String(attrVal) === String(selectedAttraction);
        });
    }, [allVersions, selectedAttraction]);

    // Lọc danh sách Parent Version cho Dropdown
    const parentVersionsList = useMemo(() => {
        return Object.entries(allVersions).filter(([vId, vObj]) => {
            if (vId === 'v0_raw') return true;
            if (selectedAttraction === 'ALL') return true;

            const attrVal = vObj?.id_attraction !== undefined ? vObj.id_attraction : vObj?.attraction_id;
            return !attrVal || attrVal === 'ALL' || String(attrVal) === String(selectedAttraction);
        });
    }, [allVersions, selectedAttraction]);

    // Fetch Statistiques từ Parent Version
    useEffect(() => {
        const fetchColumnsAndStats = async () => {
            setFetchingStats(true);
            try {
                const url = `${API_BASE}/deduplication/versions/${selectedParent}/stats?id_attraction=${selectedAttraction}`;
                const res = await fetch(url);

                if (res.ok) {
                    const data = await res.json();

                    const cols = data.available_columns || [];
                    setAvailableColumns(cols);
                    setTotalDuplicates(data.total_duplicates || 0);
                    setTotalRecords(data.total_records || 0);

                    if (cols.length > 0) {
                        setSelectedColumns(prev => prev.length === 0 ? cols : prev);
                    }

                    // Cập nhật danh sách attractions khả dụng khi chọn ALL
                    const attrs = data.available_attractions || [];
                    if (selectedAttraction === 'ALL' && Array.isArray(attrs) && attrs.length > 0) {
                        setAvailableAttractions(prev => {
                            const merged = Array.from(new Set([...prev, ...attrs]));
                            return merged.sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
                        });
                    }
                }
            } catch (err) {
                console.error('Erreur lors du chargement des colonnes et stats:', err);
            } finally {
                setFetchingStats(false);
            }
        };

        if (selectedParent) {
            fetchColumnsAndStats();
        }
    }, [selectedParent, selectedAttraction]);

    // Tự động gom id_attraction từ tất cả phiên bản hiện có
    useEffect(() => {
        if (!allVersions || Object.keys(allVersions).length === 0) return;

        const extractedAttrs = new Set();
        Object.values(allVersions).forEach(v => {
            const attrVal = v?.id_attraction !== undefined ? v.id_attraction : v?.attraction_id;
            if (attrVal && attrVal !== 'ALL') {
                extractedAttrs.add(String(attrVal));
            }
        });

        if (extractedAttrs.size > 0) {
            setAvailableAttractions(prev => {
                const combined = Array.from(new Set([...prev, ...extractedAttrs]));
                return combined.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
            });
        }
    }, [allVersions]);

    // Filtered Columns cho thanh tìm kiếm
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

    // Thực thi Deduplication
    const handleExecute = async () => {
        if (selectedColumns.length === 0) {
            setErrorMsg('Veuillez sélectionner au moins une colonne clé pour la déduplication.');
            return;
        }

        setLoading(true);
        setErrorMsg(null);

        const payload = {
            parent_version_id: selectedParent,
            id_attraction: selectedAttraction,
            subset_columns: selectedColumns.length === availableColumns.length ? [] : selectedColumns,
            keep: keepOption,
        };

        try {
            const res = await fetch(`${API_BASE}/deduplication`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Une erreur est survenue lors de la suppression des doublons.');
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

        checkboxContainer: {
            maxHeight: '220px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '8px', backgroundColor: '#f8fafc', display: 'flex', flexDirection: 'column', gap: '6px'
        },
        checkboxItem: (isChecked) => ({
            display: 'flex', alignItems: 'center', padding: '8px 12px', borderRadius: '12px', cursor: 'pointer', fontSize: '13px',
            border: '1px solid', borderColor: isChecked ? '#ef4444' : 'transparent', backgroundColor: isChecked ? '#fef2f2' : '#ffffff',
            transition: 'all 0.15s ease', gap: '8px'
        }),

        actionBtn: { width: '100%', padding: '12px 18px', backgroundColor: '#ef4444', color: '#ffffff', fontWeight: '700', border: 'none', borderRadius: '16px', cursor: 'pointer', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', transition: 'background-color 0.2s', marginTop: '8px' },
        smallBtn: (active) => ({
            padding: '4px 8px', fontSize: '11px', fontWeight: 'bold',
            backgroundColor: active ? '#fef2f2' : '#ffffff',
            color: active ? '#dc2626' : '#475569',
            border: '1px solid', borderColor: active ? '#fecaca' : '#e2e8f0',
            borderRadius: '8px', cursor: 'pointer', transition: 'all 0.15s ease'
        }),

        kpiGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '12px', marginBottom: '16px' },
        kpiCard: { backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '4px' },
        kpiTitle: { fontSize: '11px', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' },
        kpiValue: { fontSize: '18px', fontWeight: '800', color: '#0f172a' },

        tabHeader: { display: 'flex', gap: '8px', borderBottom: '1px solid #e2e8f0', paddingBottom: '10px', marginBottom: '16px' },
        tabBtn: (isActive) => ({
            padding: '6px 14px', fontSize: '12px', fontWeight: '700', borderRadius: '12px', cursor: 'pointer', border: 'none',
            backgroundColor: isActive ? '#ef4444' : '#f1f5f9', color: isActive ? '#ffffff' : '#64748b', transition: 'all 0.15s ease', display: 'flex', alignItems: 'center', gap: '6px'
        }),

        resultCard: { backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '24px', padding: '20px', minWidth: 0 },
        table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0, backgroundColor: '#ffffff', borderRadius: '16px', overflow: 'hidden', fontSize: '13px', border: '1px solid #e2e8f0' },
        th: { padding: '10px 14px', textAlign: 'left', color: '#475569', fontWeight: '700', borderBottom: '1px solid #e2e8f0', backgroundColor: '#f8fafc' },
        td: { padding: '10px 14px', borderBottom: '1px solid #f1f5f9' },

        versionItem: (isSelected) => ({
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px', borderRadius: '16px', cursor: 'pointer',
            border: '1px solid', borderColor: isSelected ? '#ef4444' : '#e2e8f0', backgroundColor: isSelected ? '#fef2f2' : '#ffffff',
            transition: 'all 0.15s ease', gap: '12px'
        })
    };

    const targetCols = useMemo(() => {
        if (!selectedVersion) return [];
        const stats = selectedVersion.stats || {};
        return selectedVersion.subset_columns || stats.key_columns || availableColumns;
    }, [selectedVersion, availableColumns]);

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* HEADER CARD */}
                <div style={styles.headerCard}>
                    <div style={styles.topBar}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={styles.iconBg('#fef2f2', '#ef4444')}>
                                <CopyX size={24} />
                            </div>
                            <div>
                                <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2' }}>
                                    Suppression des Doublons (Deduplication)
                                </h1>
                                <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', lineHeight: '1.2' }}>
                                    Détection et élimination des lignes dupliquées selon un sous-ensemble de colonnes
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
                            <CopyX size={18} color="#ef4444" /> Configuration de la Déduplication
                        </h2>

                        {/* BANNIÈRE STATISTIQUES RAPIDES */}
                        <div style={{ backgroundColor: totalDuplicates > 0 ? '#fef2f2' : '#f0fdf4', border: '1px solid', borderColor: totalDuplicates > 0 ? '#fecaca' : '#bbf7d0', padding: '12px 16px', borderRadius: '16px', marginBottom: '18px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontSize: '13px', fontWeight: '600', color: totalDuplicates > 0 ? '#991b1b' : '#166534' }}>
                                {totalDuplicates > 0 ? `⚠️ ${totalDuplicates} doublon(s) détecté(s)` : '✓ Aucun doublon détecté'}
                            </span>
                            <span style={{ fontSize: '11px', fontWeight: '700', color: '#64748b' }}>
                                Total: {totalRecords} lignes
                            </span>
                        </div>

                        {/* 1. Scope Attraction */}
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

                        {/* 2. Version Source (Parent) */}
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

                        {/* 3. Sélection des Colonnes Clefs */}
                        <div style={styles.formGroup}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                <label style={{ ...styles.label, margin: 0 }}>
                                    3. Colonnes Clefs ({selectedColumns.length}/{availableColumns.length})
                                </label>
                                <div style={{ display: 'flex', gap: '4px' }}>
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
                                        <RefreshCw className="animate-spin" size={14} /> Chargement des colonnes...
                                    </div>
                                ) : filteredColumns.length === 0 ? (
                                    <div style={{ fontSize: '12px', color: '#94a3b8', padding: '8px', textAlign: 'center' }}>
                                        Aucune colonne disponible
                                    </div>
                                ) : (
                                    filteredColumns.map((col) => {
                                        const isChecked = selectedColumns.includes(col);
                                        return (
                                            <div key={col} onClick={() => handleToggleColumn(col)} style={styles.checkboxItem(isChecked)}>
                                                {isChecked ? <CheckSquare size={16} color="#ef4444" /> : <Square size={16} color="#94a3b8" />}
                                                <span style={{ fontWeight: isChecked ? '700' : '500', color: '#1e293b', wordBreak: 'break-all', flex: 1 }}>
                                                    {col}
                                                </span>
                                            </div>
                                        );
                                    })
                                )}
                            </div>
                        </div>

                        {/* 4. Stratégie de Conservation */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>4. Stratégie de Conservation</label>
                            <div style={styles.selectBox}>
                                <select value={keepOption} onChange={(e) => setKeepOption(e.target.value)} style={styles.select}>
                                    {KEEP_OPTIONS.map((opt) => (
                                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        <button onClick={handleExecute} disabled={loading} style={{ ...styles.actionBtn, backgroundColor: loading ? '#94a3b8' : '#ef4444' }}>
                            {loading ? <RefreshCw className="animate-spin" size={16} /> : <Play size={16} />}
                            {loading ? 'Suppression dans PostgreSQL...' : `Lancer la Déduplication (${selectedColumns.length})`}
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
                                            Rapport de Déduplication
                                        </span>
                                    </div>
                                    <button onClick={() => setSelectedVersion(null)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}>
                                        <X size={18} />
                                    </button>
                                </div>

                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
                                    <span style={{ backgroundColor: '#fecaca', color: '#991b1b', fontFamily: 'monospace', fontSize: '12px', padding: '4px 8px', borderRadius: '8px', fontWeight: 'bold' }}>
                                        {selectedVersion.version_id}
                                    </span>
                                    <span style={{ backgroundColor: '#e0f2fe', color: '#0369a1', fontFamily: 'monospace', fontSize: '12px', padding: '4px 8px', borderRadius: '8px', fontWeight: 'bold' }}>
                                        Attraction: {selectedVersion.id_attraction || 'ALL'}
                                    </span>
                                </div>

                                <div style={styles.kpiGrid}>
                                    <div style={styles.kpiCard}>
                                        <span style={styles.kpiTitle}>Lignes Avant</span>
                                        <span style={styles.kpiValue}>
                                            {selectedVersion.stats?.rows_before?.toLocaleString() || 'N/A'}
                                        </span>
                                    </div>
                                    <div style={styles.kpiCard}>
                                        <span style={styles.kpiTitle}>Lignes Après</span>
                                        <span style={{ ...styles.kpiValue, color: '#166534' }}>
                                            {selectedVersion.stats?.rows_after?.toLocaleString() || 'N/A'}
                                        </span>
                                    </div>
                                    <div style={styles.kpiCard}>
                                        <span style={styles.kpiTitle}>Doublons Supprimés</span>
                                        <span style={{ ...styles.kpiValue, color: '#dc2626' }}>
                                            {selectedVersion.stats?.duplicates_removed?.toLocaleString() || '0'}
                                        </span>
                                    </div>
                                </div>

                                <div style={styles.tabHeader}>
                                    <button onClick={() => setActiveTab('summary')} style={styles.tabBtn(activeTab === 'summary')}>
                                        <Table size={14} /> Colonnes clés ({targetCols.length})
                                    </button>
                                    <button onClick={() => setActiveTab('columns')} style={styles.tabBtn(activeTab === 'columns')}>
                                        <BarChart2 size={14} /> Aperçu de la Table SQL
                                    </button>
                                </div>

                                {activeTab === 'summary' && (
                                    <div style={{ overflowX: 'auto' }}>
                                        <table style={styles.table}>
                                            <thead>
                                                <tr>
                                                    <th style={styles.th}>Colonne Clé</th>
                                                    <th style={{ ...styles.th, textAlign: 'center' }}>Statut</th>
                                                    <th style={{ ...styles.th, textAlign: 'center' }}>Règle de conservation</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {targetCols.length === 0 ? (
                                                    <tr>
                                                        <td colSpan="3" style={{ ...styles.td, textAlign: 'center', color: '#94a3b8' }}>
                                                            Toutes les colonnes ont été incluses pour vérifier les doublons.
                                                        </td>
                                                    </tr>
                                                ) : (
                                                    targetCols.map((col) => (
                                                        <tr key={col}>
                                                            <td style={styles.td}>
                                                                <code style={{ fontWeight: '700', color: '#0f172a' }}>{col}</code>
                                                            </td>
                                                            <td style={{ ...styles.td, textAlign: 'center' }}>
                                                                <span style={{ backgroundColor: '#dcfce7', color: '#15803d', fontSize: '11px', fontWeight: '800', padding: '2px 8px', borderRadius: '6px' }}>
                                                                    ✓ Vérifiée
                                                                </span>
                                                            </td>
                                                            <td style={{ ...styles.td, textAlign: 'center', color: '#ef4444', fontWeight: '800', fontSize: '12px' }}>
                                                                {selectedVersion.keep ? `Keep ${selectedVersion.keep}` : 'Conserver première'}
                                                            </td>
                                                        </tr>
                                                    ))
                                                )}
                                            </tbody>
                                        </table>
                                    </div>
                                )}

                                {activeTab === 'columns' && (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                        <div style={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', padding: '12px 16px', borderRadius: '14px', fontSize: '12px' }}>
                                            <div style={{ fontWeight: '700', color: '#0f172a', marginBottom: '6px' }}>
                                                📄 Table SQL Générée dans PostgreSQL
                                            </div>
                                            <code style={{ backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '6px 10px', borderRadius: '8px', display: 'block', color: '#ef4444', fontWeight: 'bold' }}>
                                                {selectedVersion.file_path || `dedup_${selectedVersion.version_id}`}
                                            </code>
                                        </div>

                                        <div style={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', padding: '12px 16px', borderRadius: '14px', fontSize: '12px', color: '#475569' }}>
                                            <div><strong>Version Parent :</strong> <code>{selectedVersion.parent_version_id}</code></div>
                                            <div style={{ marginTop: '4px' }}><strong>Méthode d'exécution :</strong> {selectedVersion.method_label_fr || selectedVersion.method}</div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div style={{ ...styles.card, borderStyle: 'dashed', textAlign: 'center', color: '#64748b', fontSize: '13px', padding: '24px' }}>
                                <Activity size={28} color="#94a3b8" style={{ marginBottom: '8px' }} />
                                <div>💡 <i>Sélectionnez une version ci-dessous pour consulter son rapport détaillé.</i></div>
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
                                        const attrValue = v.id_attraction !== undefined ? v.id_attraction : v.attraction_id;
                                        return (
                                            <div key={v.version_id} onClick={() => setSelectedVersion(v)} style={styles.versionItem(isSelected)}>
                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                                                        <span style={{ fontFamily: 'monospace', fontWeight: '800', color: '#ef4444', fontSize: '13px' }}>
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