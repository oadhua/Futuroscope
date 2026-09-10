import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
    Sliders, Filter, Layers, RefreshCw, AlertTriangle, CheckCircle2,
    BarChart2, Play, Trash2, Target, SlidersHorizontal, Save, CheckSquare, Square
} from 'lucide-react';

export default function FeatureSelectionModule() {
    // États de sélection des paramètres
    const [versions, setVersions] = useState({});
    const [selectedVersionId, setSelectedVersionId] = useState('');
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [targetColumn, setTargetColumn] = useState('visitor_count');
    const [selectionMethod, setSelectionMethod] = useState('random_forest');

    // Nouveaux états pour la sauvegarde
    const [customVersionId, setCustomVersionId] = useState('');
    const [selectedFeatures, setSelectedFeatures] = useState([]);
    const [saving, setSaving] = useState(false);
    const [saveSuccessMsg, setSaveSuccessMsg] = useState(null);

    // Toutes les colonnes de la table
    const [allTableColumns, setAllTableColumns] = useState([]);
    const [rawFeatureResults, setRawFeatureResults] = useState(null);

    // États de filtrage côté Frontend
    const [topKSlider, setTopKSlider] = useState(10);
    const [minScoreThreshold, setMinScoreThreshold] = useState(0);

    // États UI
    const [loading, setLoading] = useState(false);
    const [loadingStats, setLoadingStats] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);

    const API_BASE = 'http://localhost:8000';

    // 1. Charger la liste des versions
    const fetchVersions = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/data-prep/versions`);
            if (res.ok) {
                const data = await res.json();
                setVersions(data || {});
            }
        } catch (err) {
            console.error('Erreur lors du chargement des versions:', err);
        }
    }, [API_BASE]);

    useEffect(() => {
        fetchVersions();
    }, [fetchVersions]);

    // Trích xuất động danh sách các Attraction hiện có từ danh sách versions
    const availableAttractionsList = useMemo(() => {
        const attrs = new Set();
        Object.values(versions).forEach(v => {
            const attrVal = v.id_attraction || v.attraction_id;
            if (attrVal && attrVal !== 'ALL') {
                attrs.add(String(attrVal));
            }
        });
        return Array.from(attrs).sort();
    }, [versions]);

    // 2. Filtrer les versions par attraction (Danh sách đầy đủ cho dropdown đầu vào)
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

    // 2b. CHỈ LỌC CÁC PHIÊN BẢN ĐƯỢC TẠO TỪ FEATURE SELECTION (Dùng cho Mục 4)
    const fsOnlyVersions = useMemo(() => {
        return availableVersions.filter(v => {
            const vId = String(v.version_id || '');
            // Kiểm tra theo ID có chuỗi '_fs_' hoặc các trường đặc trưng của Feature Selection
            return (
                vId.includes('_fs_')
            );
        });
    }, [availableVersions]);

    useEffect(() => {
        if (availableVersions.length > 0) {
            const exists = availableVersions.some(v => v.version_id === selectedVersionId);
            if (!exists) {
                setSelectedVersionId(availableVersions[0].version_id);
            }
        } else {
            setSelectedVersionId('');
        }
    }, [availableVersions, selectedVersionId]);

    // Tự động gợi ý tên phiên bản tối giản: target_attraction_fs_index
    useEffect(() => {
        if (!selectedVersionId || !targetColumn) {
            setCustomVersionId('');
            return;
        }

        // 1. Chuẩn hóa tên Target và Attraction
        const cleanTarget = targetColumn.toLowerCase().replace(/[^a-z0-9_]/g, '');
        const attrCode = selectedAttraction === 'ALL' ? 'attrALL' : `${selectedAttraction}`;

        // 2. Định dạng chuỗi tiền tố: target_attraction_fs_
        const prefix = `${cleanTarget}_${attrCode}_fs_`;

        // 3. Tìm index lớn nhất hiện có trùng tiền tố này để tự động tăng 1
        const existingSuffixes = Object.values(versions)
            .map(v => v.version_id)
            .filter(id => id && id.startsWith(prefix))
            .map(id => parseInt(id.replace(prefix, ''), 10))
            .filter(num => !isNaN(num));

        const maxSuffix = existingSuffixes.length > 0 ? Math.max(...existingSuffixes) : 0;

        // 4. Gán tên gợi ý hoàn chỉnh
        setCustomVersionId(`${prefix}${maxSuffix + 1}`);

    }, [selectedVersionId, selectedAttraction, targetColumn, versions]);

    // 3. Récupérer toutes les colonnes de la version sélectionnée
    useEffect(() => {
        if (!selectedVersionId) {
            setAllTableColumns([]);
            setTargetColumn('');
            return;
        }

        const fetchVersionColumns = async () => {
            setLoadingStats(true);
            try {
                const attrParam = selectedAttraction ? `?id_attraction=${encodeURIComponent(selectedAttraction)}` : '?id_attraction=ALL';
                const url = `${API_BASE}/data-prep/feature-selection/versions/${selectedVersionId}/stats${attrParam}`;

                const res = await fetch(url);

                if (res.ok) {
                    const data = await res.json();
                    let cols = [];

                    if (data.available_columns && Array.isArray(data.available_columns)) {
                        cols = data.available_columns;
                    } else if (data.null_counts && typeof data.null_counts === 'object') {
                        cols = Object.keys(data.null_counts);
                    } else if (data.numeric_columns && Array.isArray(data.numeric_columns)) {
                        cols = data.numeric_columns;
                    }

                    setAllTableColumns(cols);

                    if (cols.length > 0) {
                        setTargetColumn(prev => (cols.includes(prev) ? prev : (cols.includes('visitor_count') ? 'visitor_count' : cols[0])));
                    } else {
                        setTargetColumn('');
                    }
                } else {
                    setAllTableColumns([]);
                    setTargetColumn('');
                }
            } catch (err) {
                console.error('Erreur lors de la récupération des colonnes:', err);
                setAllTableColumns([]);
                setTargetColumn('');
            } finally {
                setLoadingStats(false);
            }
        };

        fetchVersionColumns();
    }, [selectedVersionId, selectedAttraction, API_BASE]);

    // 4. Lancer le calcul d'importance
    const handleRunFeatureSelection = async () => {
        if (!selectedVersionId || !targetColumn) {
            setErrorMsg('Veuillez sélectionner une version et une variable cible.');
            return;
        }

        setLoading(true);
        setErrorMsg(null);
        setSaveSuccessMsg(null);

        try {
            const payload = {
                version_id: selectedVersionId,
                id_attraction: selectedAttraction,
                target_column: targetColumn,
                method: selectionMethod,
                calculate_all: true,
                top_k: null
            };

            const res = await fetch(`${API_BASE}/data-prep/feature-selection/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Erreur lors du calcul.');
            }

            const data = await res.json();
            setRawFeatureResults(data);

            if (data.features && data.features.length > 0) {
                const totalCount = data.features.length;
                setTopKSlider(totalCount);
                setSelectedFeatures(data.features.map(f => f.name));
            }
        } catch (err) {
            setErrorMsg(err.message);
            setRawFeatureResults(null);
        } finally {
            setLoading(false);
        }
    };

    // 5. Sauvegarder la nouvelle version filtrée
    const handleSaveNewVersion = async () => {
        if (selectedFeatures.length === 0) {
            setErrorMsg('Veuillez sélectionner au moins une caractéristique à conserver.');
            return;
        }

        setSaving(true);
        setErrorMsg(null);
        setSaveSuccessMsg(null);

        try {
            // Lấy thông tin id_attraction thực tế từ version cha đang được chọn
            const parentVersion = versions[selectedVersionId];
            const actualAttraction = parentVersion?.id_attraction || parentVersion?.attraction_id || selectedAttraction;

            const payload = {
                parent_version_id: selectedVersionId,
                id_attraction: actualAttraction, // <--- Đã sửa: lấy id_attraction gốc của parent version
                target_column: targetColumn,
                selected_features: selectedFeatures,
                method: selectionMethod,
                custom_version_id: customVersionId.trim() || null
            };

            const res = await fetch(`${API_BASE}/data-prep/feature-selection/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Erreur lors de la sauvegarde.');
            }

            const data = await res.json();
            setSaveSuccessMsg(`Nouvelle version créée avec succès : ${data.version_id}`);
            fetchVersions();
        } catch (err) {
            setErrorMsg(err.message);
        } finally {
            setSaving(false);
        }
    };

    // 6. Filtrage dynamique par Slider et Score Min
    const filteredFeatures = useMemo(() => {
        if (!rawFeatureResults || !rawFeatureResults.features) return [];

        let list = [...rawFeatureResults.features];

        if (minScoreThreshold > 0) {
            list = list.filter(f => f.score >= minScoreThreshold);
        }

        return list.slice(0, topKSlider);
    }, [rawFeatureResults, topKSlider, minScoreThreshold]);

    // Xử lý sự kiện kéo Slider mà KHÔNG làm mất các tích chọn thủ công
    const handleSliderChange = (newTopK) => {
        setTopKSlider(newTopK);
        if (rawFeatureResults?.features) {
            let list = [...rawFeatureResults.features];
            if (minScoreThreshold > 0) {
                list = list.filter(f => f.score >= minScoreThreshold);
            }
            const newTopFeatures = list.slice(0, newTopK).map(f => f.name);
            setSelectedFeatures(newTopFeatures);
        }
    };

    const toggleFeatureSelection = (name) => {
        setSelectedFeatures(prev =>
            prev.includes(name) ? prev.filter(item => item !== name) : [...prev, name]
        );
    };

    const toggleSelectAll = () => {
        if (selectedFeatures.length === filteredFeatures.length) {
            setSelectedFeatures([]);
        } else {
            setSelectedFeatures(filteredFeatures.map(f => f.name));
        }
    };

    // 7. Supprimer une version
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
                setRawFeatureResults(null);
            }
            fetchVersions();
        } catch (err) {
            alert(err.message);
        }
    };

    const totalFeaturesCount = rawFeatureResults?.features?.length || 0;
    const textBreakStyle = { wordBreak: 'break-all', overflowWrap: 'anywhere', whiteSpace: 'normal', maxWidth: '100%' };

    const styles = {
        wrapper: { minHeight: '100vh', backgroundColor: '#f8fafc', padding: '24px', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1e293b', boxSizing: 'border-box' },
        container: { maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px', width: '100%', boxSizing: 'border-box' },
        card: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)', boxSizing: 'border-box', width: '100%' },
        filterRow: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px', marginBottom: '16px' },
        formGroup: { width: '100%', boxSizing: 'border-box' },
        label: { display: 'block', fontSize: '12px', fontWeight: '700', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.025em', marginBottom: '8px' },
        selectBox: { display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '8px 14px', borderRadius: '16px', fontSize: '13px', width: '100%', boxSizing: 'border-box' },
        select: { border: 'none', background: 'transparent', fontWeight: 'bold', color: '#0f172a', outline: 'none', cursor: 'pointer', width: '100%', fontSize: '13px', ...textBreakStyle },
        input: { width: '100%', padding: '8px 14px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '14px', fontSize: '13px', fontWeight: '600', color: '#0f172a', outline: 'none', boxSizing: 'border-box' },
        actionBtn: { width: '100%', padding: '12px 18px', backgroundColor: '#2563eb', color: '#ffffff', fontWeight: '700', border: 'none', borderRadius: '16px', cursor: 'pointer', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' },
        saveBtn: { padding: '10px 18px', backgroundColor: '#16a34a', color: '#ffffff', fontWeight: '700', border: 'none', borderRadius: '14px', cursor: 'pointer', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' },
        resultCard: { backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '24px', padding: '24px', width: '100%', boxSizing: 'border-box' },
        tableWrapper: { width: '100%', overflowX: 'auto', borderRadius: '16px', border: '1px solid #e2e8f0', marginBottom: '20px', backgroundColor: '#ffffff' },
        table: { width: '100%', minWidth: '600px', borderCollapse: 'separate', borderSpacing: 0, fontSize: '13px' },
        th: { padding: '12px 16px', textAlign: 'left', color: '#475569', fontWeight: '700', borderBottom: '1px solid #e2e8f0', backgroundColor: '#f8fafc' },
        td: { padding: '12px 16px', borderBottom: '1px solid #f1f5f9', ...textBreakStyle },
        versionItem: (isSelected) => ({
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px', borderRadius: '16px', cursor: 'pointer',
            border: '1px solid', borderColor: isSelected ? '#2563eb' : '#e2e8f0', backgroundColor: isSelected ? '#f0f6ff' : '#ffffff',
            gap: '16px', boxSizing: 'border-box', width: '100%'
        })
    };

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* 1. HEADER */}
                <div style={styles.card}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        <div style={{ backgroundColor: '#eff6ff', color: '#2563eb', padding: '12px', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                            <Sliders size={28} />
                        </div>
                        <div>
                            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2', ...textBreakStyle }}>
                                Sélection et Importance des Caractéristiques
                            </h1>
                            <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: '#64748b', fontWeight: '600', ...textBreakStyle }}>
                                Analyse globale des colonnes ({allTableColumns.length} colonnes détectées)
                            </p>
                        </div>
                    </div>
                </div>

                {/* 2. CONFIGURATION */}
                <div style={styles.card}>
                    <h2 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: '700', color: '#0f172a', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' }}>
                        ⚙️ Configuration du Calcul Global
                    </h2>

                    <div style={styles.filterRow}>
                        {/* Périmètre Attraction */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>1. Périmètre Attraction</label>
                            <div style={{ ...styles.selectBox, backgroundColor: '#f0f9ff', borderColor: '#bae6fd' }}>
                                <Filter size={16} color="#0284c7" style={{ flexShrink: 0 }} />
                                <select
                                    value={selectedAttraction}
                                    onChange={(e) => setSelectedAttraction(e.target.value)}
                                    style={{ ...styles.select, color: '#0369a1' }}
                                >
                                    <option value="ALL">🌐 Tous les sites (ALL)</option>
                                    {availableAttractionsList.map((attr) => (
                                        <option key={attr} value={attr}>🎢 Attraction ID: {attr}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* Version à Analyser */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>2. Version des Données</label>
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

                        {/* Variable Cible */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>
                                3. Variable Cible ({allTableColumns.length} Colonnes)
                            </label>
                            <div style={styles.selectBox}>
                                <Target size={16} color="#94a3b8" style={{ flexShrink: 0 }} />
                                {loadingStats ? (
                                    <span style={{ fontSize: '12px', color: '#64748b' }}>Chargement...</span>
                                ) : (
                                    <select
                                        value={targetColumn}
                                        onChange={(e) => setTargetColumn(e.target.value)}
                                        style={styles.select}
                                    >
                                        {allTableColumns
                                            .filter(col => {
                                                const lowerCol = col.toLowerCase();
                                                return lowerCol === 'visitor_count' || lowerCol.includes('elec') || lowerCol.includes('ec');
                                            })
                                            .map((col) => (
                                                <option key={col} value={col}>{col}</option>
                                            ))
                                        }
                                    </select>
                                )}
                            </div>
                        </div>

                        {/* Méthode d'évaluation */}
                        <div style={styles.formGroup}>
                            <label style={styles.label}>4. Méthode d'Évaluation</label>
                            <div style={styles.selectBox}>
                                <BarChart2 size={16} color="#94a3b8" style={{ flexShrink: 0 }} />
                                <select
                                    value={selectionMethod}
                                    onChange={(e) => setSelectionMethod(e.target.value)}
                                    style={styles.select}
                                >
                                    <option value="random_forest">🌲 Random Forest Importance</option>
                                    <option value="xgboost">⚡ XGBoost Feature Importance</option>
                                    <option value="shap">🔍 SHAP Values</option>
                                    <option value="correlation">📊 Corrélation de Pearson</option>
                                    <option value="mutual_info">💡 Information Mutuelle</option>
                                    <option value="lasso">🎯 Régularisation Lasso (L1)</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={handleRunFeatureSelection}
                        disabled={loading || !selectedVersionId || !targetColumn}
                        style={{ ...styles.actionBtn, backgroundColor: loading ? '#94a3b8' : '#2563eb' }}
                    >
                        {loading ? <RefreshCw className="animate-spin" size={18} /> : <Play size={18} />}
                        {loading ? 'Calcul global en cours...' : `Calculer l'Importance des Caractéristiques`}
                    </button>

                    {errorMsg && (
                        <div style={{ marginTop: '16px', padding: '14px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '16px', color: '#991b1b', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <AlertTriangle size={18} color="#dc2626" style={{ flexShrink: 0 }} />
                            <span style={textBreakStyle}>{errorMsg}</span>
                        </div>
                    )}
                </div>

                {/* 3. RAPPORT & SAUVEGARDE */}
                {rawFeatureResults && (
                    <div style={styles.resultCard}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <CheckCircle2 size={22} color="#1e40af" style={{ flexShrink: 0 }} />
                                <h2 style={{ margin: 0, fontWeight: '800', color: '#1e40af', fontSize: '16px', ...textBreakStyle }}>
                                    Résultats de l'Analyse des Caractéristiques
                                </h2>
                            </div>
                        </div>

                        {/* CURSEUR DE SÉLECTION DU NOMBRE DE VARIABLES (TOP K) */}
                        <div style={{ backgroundColor: '#ffffff', padding: '20px', borderRadius: '16px', border: '1px solid #bfdbfe', marginBottom: '20px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', fontWeight: '800', color: '#1e40af', marginBottom: '16px' }}>
                                <SlidersHorizontal size={18} />
                                <span>Sélection du Nombre de Variables (Top {topKSlider} / {totalFeaturesCount})</span>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', alignItems: 'center' }}>
                                {/* Curseur Top K */}
                                <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: '700', color: '#334155', marginBottom: '8px' }}>
                                        <span>Nombre de variables souhaité :</span>
                                        <span style={{ backgroundColor: '#dbeafe', color: '#1e40af', padding: '2px 10px', borderRadius: '12px', fontSize: '14px' }}>
                                            {topKSlider} variables
                                        </span>
                                    </div>
                                    <input
                                        type="range"
                                        min={1}
                                        max={totalFeaturesCount || 1}
                                        value={topKSlider}
                                        onChange={(e) => handleSliderChange(parseInt(e.target.value, 10))}
                                        style={{ width: '100%', cursor: 'pointer', accentColor: '#2563eb' }}
                                    />
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8', marginTop: '4px' }}>
                                        <span>1</span>
                                        <span>{totalFeaturesCount} variables</span>
                                    </div>
                                </div>

                                {/* Seuil de Score Min */}
                                <div>
                                    <label style={styles.label}>Score minimal ({">="} {minScoreThreshold})</label>
                                    <input
                                        type="number"
                                        step="0.001"
                                        min="0"
                                        max="1"
                                        value={minScoreThreshold}
                                        onChange={(e) => setMinScoreThreshold(parseFloat(e.target.value) || 0)}
                                        style={styles.input}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* TABLEAU DES CARACTÉRISTIQUES SÉLECTIONNABLES */}
                        <div style={styles.tableWrapper}>
                            <table style={styles.table}>
                                <thead>
                                    <tr>
                                        <th style={{ ...styles.th, width: '40px' }}>
                                            <button onClick={toggleSelectAll} style={{ background: 'none', border: 'none', cursor: 'pointer' }}>
                                                {selectedFeatures.length === filteredFeatures.length && filteredFeatures.length > 0 ? <CheckSquare size={18} color="#2563eb" /> : <Square size={18} color="#94a3b8" />}
                                            </button>
                                        </th>
                                        <th style={styles.th}>Rang</th>
                                        <th style={styles.th}>Nom de la Variable (Feature)</th>
                                        <th style={{ ...styles.th, textAlign: 'center' }}>Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredFeatures.map((f, idx) => {
                                        const isChecked = selectedFeatures.includes(f.name);
                                        return (
                                            <tr key={f.name} onClick={() => toggleFeatureSelection(f.name)} style={{ cursor: 'pointer', backgroundColor: isChecked ? '#f0fdf4' : 'transparent' }}>
                                                <td style={styles.td}>
                                                    {isChecked ? <CheckSquare size={18} color="#16a34a" /> : <Square size={18} color="#94a3b8" />}
                                                </td>
                                                <td style={{ ...styles.td, fontWeight: '700' }}>#{idx + 1}</td>
                                                <td style={{ ...styles.td, fontFamily: 'monospace', color: '#2563eb', fontWeight: '600' }}>{f.name}</td>
                                                <td style={{ ...styles.td, textAlign: 'center', fontWeight: '700' }}>
                                                    {typeof f.score === 'number' ? f.score.toFixed(4) : f.score}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>

                        {/* ZONE DE SAUVEGARDE DE LA NOUVELLE VERSION */}
                        <div style={{ backgroundColor: '#ffffff', padding: '20px', borderRadius: '16px', border: '1px solid #bbf7d0' }}>
                            <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: '800', color: '#166534' }}>
                                💾 Sauvegarder la Version Filtrée ({selectedFeatures.length} caractéristiques sélectionnées)
                            </h3>
                            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                                <input
                                    type="text"
                                    placeholder="Nom personnalisé (Optionnel, ex: v1_features_h03)"
                                    value={customVersionId}
                                    onChange={(e) => setCustomVersionId(e.target.value)}
                                    style={{ ...styles.input, flex: 1, minWidth: '200px' }}
                                />
                                <button
                                    onClick={handleSaveNewVersion}
                                    disabled={saving || selectedFeatures.length === 0}
                                    style={{ ...styles.saveBtn, backgroundColor: saving ? '#94a3b8' : '#16a34a' }}
                                >
                                    {saving ? <RefreshCw className="animate-spin" size={16} /> : <Save size={16} />}
                                    {saving ? 'Sauvegarde...' : 'Créer la Nouvelle Version'}
                                </button>
                            </div>
                            {saveSuccessMsg && (
                                <p style={{ marginTop: '10px', fontSize: '13px', color: '#15803d', fontWeight: '700', margin: '10px 0 0 0' }}>
                                    ✅ {saveSuccessMsg}
                                </p>
                            )}
                        </div>

                    </div>
                )}

                {/* 4. HISTORIQUE DES VERSIONS */}
                <div style={styles.card}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
                        <h2 style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#0f172a' }}>
                            📂 Versions Disponibles ({fsOnlyVersions.length})
                        </h2>
                    </div>

                    {fsOnlyVersions.length === 0 ? (
                        <p style={{ color: '#94a3b8', fontStyle: 'italic', fontSize: '13px', margin: 0 }}>
                            Aucune version issue de Feature Selection disponible.
                        </p>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', width: '100%' }}>
                            {fsOnlyVersions.map((v) => {
                                const isSelected = selectedVersionId === v.version_id;
                                return (
                                    <div key={v.version_id} onClick={() => setSelectedVersionId(v.version_id)} style={styles.versionItem(isSelected)}>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                                <span style={{ fontFamily: 'monospace', fontWeight: '800', color: '#2563eb', fontSize: '14px', ...textBreakStyle }}>
                                                    {v.version_id}
                                                </span>
                                                <span style={{ backgroundColor: '#f1f5f9', color: '#475569', fontSize: '11px', padding: '3px 8px', borderRadius: '6px', fontWeight: '600' }}>
                                                    {v.method_label_fr || v.method || 'Feature Selection'}
                                                </span>
                                            </div>
                                        </div>
                                        <button onClick={(e) => handleDeleteVersion(e, v.version_id)} style={{ background: 'none', border: 'none', color: '#dc2626', cursor: 'pointer', padding: '6px' }}>
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