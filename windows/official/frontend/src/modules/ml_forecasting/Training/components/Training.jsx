import React, { useState, useEffect } from 'react';

const API_ML_BASE = 'http://localhost:8000/ml/training';

export default function MLTrainingDashboard() {
    // ==================== ÉTATS DES DONNÉES ====================
    const [attractions, setAttractions] = useState([]);
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [versions, setVersions] = useState([]);
    const [selectedVersionId, setSelectedVersionId] = useState('');

    const [availableColumns, setAvailableColumns] = useState([]);
    const [targetColumn, setTargetColumn] = useState('visitor_count');
    // targetType: 'visitor' | 'elec' | 'thermal'
    const [targetType, setTargetType] = useState('visitor');
    const [modelType, setModelType] = useState('xgboost');

    // Configuration Train/Test Split
    const [splitMethod, setSplitMethod] = useState('ratio'); // 'ratio' | 'date'
    const [testSize, setTestSize] = useState(0.2);

    // Date ranges
    const [trainStartDate, setTrainStartDate] = useState('');
    const [trainEndDate, setTrainEndDate] = useState('');
    const [testStartDate, setTestStartDate] = useState('');
    const [testEndDate, setTestEndDate] = useState('');

    // ==================== HYPERPARAMÈTRES ====================
    const [xgbParams, setXgbParams] = useState({ n_estimators: 100, learning_rate: 0.05, max_depth: 6, subsample: 1, colsample_bytree: 1, random_state: 42 });
    const [lgbParams, setLgbParams] = useState({ n_estimators: 100, learning_rate: 0.05, num_leaves: 31, max_depth: -1, random_state: 42 });
    const [rfParams, setRfParams] = useState({ n_estimators: 100, max_depth: 12, min_samples_split: 2, min_samples_leaf: 1, random_state: 42 });
    const [ridgeParams, setRidgeParams] = useState({ alpha: 1.0, solver: 'auto' });
    const [rnnParams, setRnnParams] = useState({ epochs: 20, batch_size: 32, hidden_dim: 64, time_steps: 12, learning_rate: 0.001, dropout: 0.2 });

    // UI States
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);
    const [successMsg, setSuccessMsg] = useState(null);
    const [trainingResult, setTrainingResult] = useState(null);
    const [modelRegistry, setModelRegistry] = useState([]);
    const [selectedHistoryModel, setSelectedHistoryModel] = useState(null);

    // 1. Tải danh sách attractions khi khởi chạy
    useEffect(() => {
        fetchAttractions();
    }, []);

    // 2. Tải danh sách version khi thay đổi attraction hoặc targetType
    useEffect(() => {
        fetchVersions(selectedAttraction);
    }, [selectedAttraction, targetType]);

    // 3. Tải danh sách cột và khoảng thời gian khi chọn version
    useEffect(() => {
        if (selectedVersionId) {
            fetchVersionColumns(selectedVersionId, selectedAttraction);
            fetchVersionDateRange(selectedVersionId, selectedAttraction);
        } else {
            setAvailableColumns([]);
        }
    }, [selectedVersionId, selectedAttraction]);

    // 4. Tải danh sách mô hình đã lưu
    useEffect(() => {
        fetchModelRegistry();
    }, [targetType]);

    // ==================== REQUÊTES API ====================

    const fetchAttractions = async () => {
        try {
            const res = await fetch(`${API_ML_BASE}/attractions`);
            if (res.ok) {
                const data = await res.json();
                const rawList = data.attractions || [];
                const list = rawList
                    .map((item) => {
                        if (typeof item === 'object' && item !== null) {
                            return item.id || item.id_attraction || item.attraction_id || item.name || String(item);
                        }
                        return String(item);
                    })
                    .filter((item) => item.toUpperCase() !== 'ALL');

                setAttractions(list);
            }
        } catch (e) {
            console.error('Erreur lors du chargement des attractions:', e);
        }
    };

    const fetchVersions = async (attractionId, targetVersionToSelect = null) => {
        try {
            const params = new URLSearchParams({ id_attraction: attractionId });
            const res = await fetch(`${API_ML_BASE}/versions?${params.toString()}`);
            if (res.ok) {
                const data = await res.json();
                let rawList = [];
                if (data && typeof data === 'object' && !Array.isArray(data) && !data.versions) {
                    rawList = Object.values(data);
                } else if (data && Array.isArray(data.versions)) {
                    rawList = data.versions;
                } else if (Array.isArray(data)) {
                    rawList = data;
                }

                const currentAttr = String(attractionId || 'ALL').trim().toUpperCase();
                const filteredList = rawList.filter((item) => {
                    const vId = String(item.version_id || '').toLowerCase();

                    if (!vId.includes('fs')) return false;

                    // === SỬA TẠI ĐÂY: Dùng Regex chính xác theo prefix/tag ===
                    if (targetType === 'visitor' && !vId.includes('visitor')) return false;
                    if (targetType === 'elec' && !/_elec_|_elec$|^elec_/.test(vId)) return false;
                    if (targetType === 'thermal' && !/_ec_|_ec$|^ec_/.test(vId)) return false;

                    if (currentAttr === 'ALL') return true;
                    const itemAttr = String(item.id_attraction || '').trim().toUpperCase();

                    const vIdUpper = vId.toUpperCase();
                    if (vIdUpper === 'V0_RAW') return true;
                    const hasOtherAttractionCode = /_H\d+/i.test(vIdUpper) && !vIdUpper.includes(`_${currentAttr}`);
                    if (hasOtherAttractionCode) return false;

                    return itemAttr === currentAttr || vIdUpper.includes(`_${currentAttr}`);
                });

                const uniqueVersionsMap = new Map();
                filteredList.forEach(item => {
                    if (item.version_id && !uniqueVersionsMap.has(item.version_id)) {
                        uniqueVersionsMap.set(item.version_id, item);
                    }
                });
                const cleanList = Array.from(uniqueVersionsMap.values());
                cleanList.sort((a, b) => (a.version_id || '').localeCompare(b.version_id || ''));

                if (cleanList.length > 0) {
                    setVersions(cleanList);
                    setSelectedVersionId((prevSelected) => {
                        const candidate = targetVersionToSelect || prevSelected;
                        const exists = cleanList.some((v) => v.version_id === candidate);
                        return exists ? candidate : cleanList[0].version_id;
                    });
                } else {
                    setVersions([]);
                    setSelectedVersionId('');
                }
            }
        } catch (e) {
            console.error('Erreur lors du chargement des versions:', e);
            setVersions([]);
            setSelectedVersionId('');
        }
    };

    const fetchVersionColumns = async (versionId, attractionId) => {
        if (!versionId) return;
        try {
            const params = new URLSearchParams({ id_attraction: attractionId });
            const res = await fetch(`${API_ML_BASE}/versions/${versionId}/columns?${params.toString()}`);
            if (res.ok) {
                const data = await res.json();
                const cols = data.columns || [];

                // Lọc bỏ các cột định danh, ngày tháng và các cột trạng thái đóng/mở cửa khỏi target mặc định
                let numericCols = cols.filter((c) => !['datetime', 'id_attraction', 'attraction_id', 'ouvert'].includes(c));

                numericCols = numericCols.filter((col) => {
                    const c = col.toLowerCase();
                    if (targetType === 'visitor') {
                        return !c.includes('elec') && !c.includes('ec_') && !c.startsWith('ec');
                    }
                    if (targetType === 'elec') {
                        return !c.includes('ec_') && !c.startsWith('ec') && !c.includes('visitor');
                    }
                    if (targetType === 'thermal') {
                        return !c.includes('elec') && !c.includes('visitor');
                    }
                    return true;
                });

                setAvailableColumns(numericCols);

                if (numericCols.length > 0) {
                    // Ưu tiên chọn cột chứa 'kwh', 'kw', 'chaleur', 'puissance', 'frequentation' làm target
                    const defaultTarget = numericCols.find(c =>
                        /kwh|kw|chaleur|puissance|visitor|frequentation/i.test(c)
                    ) || numericCols[0];

                    setTargetColumn((prev) => (numericCols.includes(prev) ? prev : defaultTarget));
                } else {
                    setTargetColumn('');
                }
            } else {
                setAvailableColumns([]);
            }
        } catch (e) {
            console.error('Erreur lors du chargement des colonnes:', e);
            setAvailableColumns([]);
        }
    };

    const fetchVersionDateRange = async (versionId, attractionId) => {
        if (!versionId) return;
        try {
            const params = new URLSearchParams({ id_attraction: attractionId });
            const res = await fetch(`${API_ML_BASE}/versions/${versionId}/date-range?${params.toString()}`);
            if (res.ok) {
                const data = await res.json();
                if (data.min_date && data.max_date) {
                    const minD = data.min_date.split('T')[0];
                    const maxD = data.max_date.split('T')[0];

                    setTrainStartDate(minD);
                    setTrainEndDate(maxD);
                    setTestStartDate(minD);
                    setTestEndDate(maxD);
                }
            }
        } catch (e) {
            console.error('Erreur lors du chargement des dates min/max:', e);
        }
    };

    const fetchModelRegistry = async () => {
        try {
            const res = await fetch(`${API_ML_BASE}/models?target_type=${targetType}`);
            if (res.ok) {
                const data = await res.json();
                if (data.models) setModelRegistry(data.models);
            }
        } catch (e) {
            console.error('Erreur lors du chargement du registre des modèles:', e);
        }
    };

    const handleDeleteVersion = async () => {
        if (!selectedVersionId || selectedVersionId === 'v0_raw') {
            setErrorMsg('Impossible de supprimer la version initiale (v0_raw).');
            return;
        }

        const confirmDelete = window.confirm(`Êtes-vous sûr de vouloir supprimer la version "${selectedVersionId}" ?`);
        if (!confirmDelete) return;

        setLoading(true);
        setErrorMsg(null);
        setSuccessMsg(null);

        try {
            const params = new URLSearchParams({
                version_id: selectedVersionId,
                id_attraction: selectedAttraction,
            });

            const res = await fetch(`${API_ML_BASE}/versions?${params.toString()}`, { method: 'DELETE' });
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Erreur lors de la suppression.');
            }

            const data = await res.json();
            setSuccessMsg(data.message || `Version ${selectedVersionId} supprimée avec succès.`);
            await fetchVersions(selectedAttraction);
        } catch (err) {
            setErrorMsg(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteModel = async (modelId) => {
        const confirmDelete = window.confirm(`Êtes-vous sûr de vouloir supprimer le modèle "${modelId}" ?`);
        if (!confirmDelete) return;

        setErrorMsg(null);
        setSuccessMsg(null);

        try {
            const res = await fetch(`${API_ML_BASE}/models/${modelId}`, { method: 'DELETE' });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Échec de la suppression du modèle.');

            setSuccessMsg(`Modèle ${modelId} supprimé avec succès.`);
            if (selectedHistoryModel?.model_id === modelId) {
                setSelectedHistoryModel(null);
            }
            fetchModelRegistry();
        } catch (err) {
            setErrorMsg(err.message);
        }
    };

    const getActiveHyperparameters = () => {
        switch (modelType) {
            case 'xgboost':
                return {
                    n_estimators: Number(xgbParams.n_estimators),
                    learning_rate: Number(xgbParams.learning_rate),
                    max_depth: Number(xgbParams.max_depth),
                    subsample: Number(xgbParams.subsample),
                    colsample_bytree: Number(xgbParams.colsample_bytree),
                    random_state: Number(xgbParams.random_state)
                };
            case 'lightgbm':
                return {
                    n_estimators: Number(lgbParams.n_estimators),
                    learning_rate: Number(lgbParams.learning_rate),
                    num_leaves: Number(lgbParams.num_leaves),
                    max_depth: Number(lgbParams.max_depth),
                    random_state: Number(lgbParams.random_state)
                };
            case 'random_forest':
                return {
                    n_estimators: Number(rfParams.n_estimators),
                    max_depth: Number(rfParams.max_depth),
                    min_samples_split: Number(rfParams.min_samples_split),
                    min_samples_leaf: Number(rfParams.min_samples_leaf),
                    random_state: Number(rfParams.random_state)
                };
            case 'ridge':
                return {
                    alpha: Number(ridgeParams.alpha),
                    solver: ridgeParams.solver
                };
            case 'lstm':
            case 'gru':
                return {
                    epochs: Number(rnnParams.epochs),
                    batch_size: Number(rnnParams.batch_size),
                    hidden_dim: Number(rnnParams.hidden_dim),
                    time_steps: Number(rnnParams.time_steps),
                    learning_rate: Number(rnnParams.learning_rate),
                    dropout: Number(rnnParams.dropout)
                };
            default:
                return {};
        }
    };

    const handleRunTraining = async (e) => {
        e.preventDefault();
        setLoading(true);
        setErrorMsg(null);
        setSuccessMsg(null);
        setTrainingResult(null);

        const payload = {
            version_id: selectedVersionId,
            id_attraction: selectedAttraction,
            target_column: targetColumn.trim(),
            target_type: targetType,
            model_type: modelType,
            split_method: splitMethod,
            test_size: Number(testSize),
            start_date: splitMethod === 'date' ? trainStartDate || null : null,
            end_date: splitMethod === 'date' ? testEndDate || null : null,
            split_date: splitMethod === 'date' ? testStartDate || null : null,
            hyperparameters: getActiveHyperparameters()
        };

        try {
            const res = await fetch(`${API_ML_BASE}/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Échec de l'entraînement du modèle.");

            setTrainingResult(data);
            setSuccessMsg(`Modèle entraîné avec succès ! ID: ${data.model_id}`);
            fetchModelRegistry();
        } catch (err) {
            setErrorMsg(err.message);
        } finally {
            setLoading(false);
        }
    };

    const renderHyperparameterInputs = () => {
        switch (modelType) {
            case 'xgboost':
                return (
                    <div style={styles.grid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Nombre d'arbres (n_estimators)</label>
                            <input type="number" value={xgbParams.n_estimators} onChange={(e) => setXgbParams({ ...xgbParams, n_estimators: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Taux d'apprentissage (learning_rate)</label>
                            <input type="number" step="0.01" value={xgbParams.learning_rate} onChange={(e) => setXgbParams({ ...xgbParams, learning_rate: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Profondeur maximale (max_depth)</label>
                            <input type="number" value={xgbParams.max_depth} onChange={(e) => setXgbParams({ ...xgbParams, max_depth: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Échantillonnage de lignes (subsample)</label>
                            <input type="number" step="0.05" value={xgbParams.subsample} onChange={(e) => setXgbParams({ ...xgbParams, subsample: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Échantillonnage de colonnes (colsample_bytree)</label>
                            <input type="number" step="0.05" value={xgbParams.colsample_bytree} onChange={(e) => setXgbParams({ ...xgbParams, colsample_bytree: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Graine aléatoire (random_state)</label>
                            <input type="number" value={xgbParams.random_state} onChange={(e) => setXgbParams({ ...xgbParams, random_state: e.target.value })} style={styles.input} />
                        </div>
                    </div>
                );

            case 'lightgbm':
                return (
                    <div style={styles.grid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Nombre d'arbres (n_estimators)</label>
                            <input type="number" value={lgbParams.n_estimators} onChange={(e) => setLgbParams({ ...lgbParams, n_estimators: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Taux d'apprentissage (learning_rate)</label>
                            <input type="number" step="0.01" value={lgbParams.learning_rate} onChange={(e) => setLgbParams({ ...lgbParams, learning_rate: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Nombre max de feuilles (num_leaves)</label>
                            <input type="number" value={lgbParams.num_leaves} onChange={(e) => setLgbParams({ ...lgbParams, num_leaves: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Profondeur max (-1 = Illimitée)</label>
                            <input type="number" value={lgbParams.max_depth} onChange={(e) => setLgbParams({ ...lgbParams, max_depth: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Graine aléatoire (random_state)</label>
                            <input type="number" value={lgbParams.random_state} onChange={(e) => setLgbParams({ ...lgbParams, random_state: e.target.value })} style={styles.input} />
                        </div>
                    </div>
                );

            case 'random_forest':
                return (
                    <div style={styles.grid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Nombre d'arbres (n_estimators)</label>
                            <input type="number" value={rfParams.n_estimators} onChange={(e) => setRfParams({ ...rfParams, n_estimators: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Profondeur maximale (max_depth)</label>
                            <input type="number" value={rfParams.max_depth} onChange={(e) => setRfParams({ ...rfParams, max_depth: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Échantillons min. par séparation (min_samples_split)</label>
                            <input type="number" value={rfParams.min_samples_split} onChange={(e) => setRfParams({ ...rfParams, min_samples_split: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Échantillons min. par feuille (min_samples_leaf)</label>
                            <input type="number" value={rfParams.min_samples_leaf} onChange={(e) => setRfParams({ ...rfParams, min_samples_leaf: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Graine aléatoire (random_state)</label>
                            <input type="number" value={rfParams.random_state} onChange={(e) => setRfParams({ ...rfParams, random_state: e.target.value })} style={styles.input} />
                        </div>
                    </div>
                );

            case 'ridge':
                return (
                    <div style={styles.grid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Pénalité de régularisation (alpha)</label>
                            <input type="number" step="0.1" value={ridgeParams.alpha} onChange={(e) => setRidgeParams({ ...ridgeParams, alpha: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Solveur d'optimisation (solver)</label>
                            <select value={ridgeParams.solver} onChange={(e) => setRidgeParams({ ...ridgeParams, solver: e.target.value })} style={styles.select}>
                                <option value="auto">auto</option>
                                <option value="svd">svd</option>
                                <option value="cholesky">cholesky</option>
                                <option value="lsqr">lsqr</option>
                                <option value="saga">saga</option>
                            </select>
                        </div>
                    </div>
                );

            case 'lstm':
            case 'gru':
                return (
                    <div style={styles.grid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Nombre d'époques (epochs)</label>
                            <input type="number" value={rnnParams.epochs} onChange={(e) => setRnnParams({ ...rnnParams, epochs: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Taille du lot (batch_size)</label>
                            <input type="number" value={rnnParams.batch_size} onChange={(e) => setRnnParams({ ...rnnParams, batch_size: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Dimension cachée (hidden_dim)</label>
                            <input type="number" value={rnnParams.hidden_dim} onChange={(e) => setRnnParams({ ...rnnParams, hidden_dim: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Pas temporels / Séquence (time_steps)</label>
                            <input type="number" value={rnnParams.time_steps} onChange={(e) => setRnnParams({ ...rnnParams, time_steps: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Taux d'apprentissage (learning_rate)</label>
                            <input type="number" step="0.0001" value={rnnParams.learning_rate} onChange={(e) => setRnnParams({ ...rnnParams, learning_rate: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Taux d'abandon (dropout)</label>
                            <input type="number" step="0.05" value={rnnParams.dropout} onChange={(e) => setRnnParams({ ...rnnParams, dropout: e.target.value })} style={styles.input} />
                        </div>
                    </div>
                );

            default:
                return null;
        }
    };

    return (
        <div style={styles.container}>
            {/* Header */}
            <div style={styles.header}>
                <h2 style={styles.title}>🧠 Studio d'Apprentissage Automatique (Entraînement des Modèles)</h2>
                <p style={styles.subtitle}>
                    Entraînement, optimisation des hyperparamètres et évaluation des modèles Machine Learning & Deep Learning.
                </p>
            </div>

            {/* Target Selector Tabs */}
            <div style={styles.tabContainer}>
                <button
                    onClick={() => setTargetType('visitor')}
                    style={targetType === 'visitor' ? styles.tabActive : styles.tabInactive}
                >
                    👥 Prédiction Fréquentation (Visiteurs)
                </button>
                <button
                    onClick={() => setTargetType('elec')}
                    style={targetType === 'elec' ? styles.tabActive : styles.tabInactive}
                >
                    ⚡ Énergie Électrique (elec)
                </button>
                <button
                    onClick={() => setTargetType('thermal')}
                    style={targetType === 'thermal' ? styles.tabActive : styles.tabInactive}
                >
                    🔥 Énergie Thermique (ec)
                </button>
            </div>

            {errorMsg && <div style={styles.errorAlert}>⚠️ {errorMsg}</div>}
            {successMsg && <div style={styles.successAlert}>✅ {successMsg}</div>}

            {/* Main Form */}
            <div style={styles.card}>
                <h3 style={styles.cardTitle}>⚙️ Configuration de l'entraînement</h3>
                <form onSubmit={handleRunTraining}>
                    <div style={styles.grid}>
                        {/* 1. Attraction */}
                        <div style={styles.fieldGroup}>
                            <label style={styles.label}>1. Identifiant de l'Attraction</label>
                            <select
                                value={selectedAttraction}
                                onChange={(e) => setSelectedAttraction(e.target.value)}
                                style={styles.select}
                            >
                                <option value="ALL">Toutes les attractions (ALL)</option>
                                {attractions.map((att) => (
                                    <option key={att} value={att}>
                                        {att}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* 2. Version */}
                        <div style={styles.fieldGroup}>
                            <label style={styles.label}>2. Version des Données (Input)</label>
                            <div style={styles.inputGroupAlign}>
                                <select
                                    value={selectedVersionId}
                                    onChange={(e) => setSelectedVersionId(e.target.value)}
                                    style={{ ...styles.select, flex: 1, minWidth: 0 }}
                                >
                                    {versions.length > 0 ? (
                                        versions.map((v) => (
                                            <option key={v.version_id} value={v.version_id}>
                                                {v.version_id}
                                            </option>
                                        ))
                                    ) : (
                                        <option value="">
                                            {targetType === 'visitor' && 'Aucune version "visitor" & "fs" disponible'}
                                            {targetType === 'elec' && 'Aucune version "elec" & "fs" disponible'}
                                            {targetType === 'thermal' && 'Aucune version "ec" & "fs" disponible'}
                                        </option>
                                    )}
                                </select>
                                <button
                                    type="button"
                                    onClick={handleDeleteVersion}
                                    disabled={loading || !selectedVersionId || selectedVersionId === 'v0_raw'}
                                    style={
                                        loading || !selectedVersionId || selectedVersionId === 'v0_raw'
                                            ? styles.buttonDeleteDisabled
                                            : styles.buttonDelete
                                    }
                                >
                                    🗑️ Supprimer
                                </button>
                            </div>
                        </div>

                        {/* 3. Target Column */}
                        <div style={styles.fieldGroup}>
                            <label style={styles.label}>3. Variable Cible (Target Column)</label>
                            {availableColumns.length > 0 ? (
                                <select
                                    value={targetColumn}
                                    onChange={(e) => setTargetColumn(e.target.value)}
                                    style={styles.select}
                                >
                                    {availableColumns.map((col) => (
                                        <option key={col} value={col}>
                                            🎯 {col}
                                        </option>
                                    ))}
                                </select>
                            ) : (
                                <input
                                    type="text"
                                    value={targetColumn}
                                    onChange={(e) => setTargetColumn(e.target.value)}
                                    placeholder="ex: visitor_count, elec_01,..."
                                    style={styles.input}
                                />
                            )}
                        </div>

                        {/* 4. Model Type */}
                        <div style={styles.fieldGroup}>
                            <label style={styles.label}>4. Algorithme de Modélisation</label>
                            <select
                                value={modelType}
                                onChange={(e) => setModelType(e.target.value)}
                                style={styles.select}
                            >
                                <optgroup label="Machine Learning (Données Tabulaires)">
                                    <option value="xgboost">XGBoost Regressor</option>
                                    <option value="lightgbm">LightGBM Regressor</option>
                                    <option value="random_forest">Random Forest Regressor</option>
                                    <option value="ridge">Ridge Regression</option>
                                </optgroup>
                                <optgroup label="Deep Learning (Séries Temporelles)">
                                    <option value="lstm">LSTM (Long Short-Term Memory)</option>
                                    <option value="gru">GRU (Gated Recurrent Unit)</option>
                                </optgroup>
                            </select>
                        </div>
                    </div>

                    {/* Train/Test Split Config */}
                    <div style={styles.splitGroup}>
                        <div style={{ display: 'flex', gap: '16px', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap' }}>
                            <label style={{ ...styles.label, marginBottom: 0 }}>Méthode de séparation (Train / Test Split) :</label>
                            <label style={styles.radioLabel}>
                                <input
                                    type="radio"
                                    name="splitMethod"
                                    value="ratio"
                                    checked={splitMethod === 'ratio'}
                                    onChange={() => setSplitMethod('ratio')}
                                /> Par Ratio (%)
                            </label>
                            <label style={styles.radioLabel}>
                                <input
                                    type="radio"
                                    name="splitMethod"
                                    value="date"
                                    checked={splitMethod === 'date'}
                                    onChange={() => setSplitMethod('date')}
                                /> Par Plage de Dates (Train / Test)
                            </label>
                        </div>

                        {splitMethod === 'ratio' ? (
                            <div>
                                <label style={styles.subLabel}>
                                    Taille du jeu de test (Test Split Ratio) : <strong>{Math.round(testSize * 100)}%</strong>
                                </label>
                                <input
                                    type="range"
                                    min="0.1"
                                    max="0.4"
                                    step="0.05"
                                    value={testSize}
                                    onChange={(e) => setTestSize(parseFloat(e.target.value))}
                                    style={styles.slider}
                                />
                            </div>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                <div style={styles.dateBox}>
                                    <h5 style={styles.dateBoxTitle}>📅 Jeu d'entraînement (Train Set)</h5>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
                                        <div style={styles.fieldGroup}>
                                            <label style={styles.subLabel}>Date de début (Start) :</label>
                                            <input
                                                type="date"
                                                value={trainStartDate}
                                                onChange={(e) => setTrainStartDate(e.target.value)}
                                                style={styles.input}
                                                required={splitMethod === 'date'}
                                            />
                                        </div>
                                        <div style={styles.fieldGroup}>
                                            <label style={styles.subLabel}>Date de fin (End) :</label>
                                            <input
                                                type="date"
                                                value={trainEndDate}
                                                onChange={(e) => setTrainEndDate(e.target.value)}
                                                style={styles.input}
                                                required={splitMethod === 'date'}
                                            />
                                        </div>
                                    </div>
                                </div>

                                <div style={styles.dateBox}>
                                    <h5 style={styles.dateBoxTitle}>🧪 Jeu de test (Test Set)</h5>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
                                        <div style={styles.fieldGroup}>
                                            <label style={styles.subLabel}>Date de bascule / Séparation (Split Date) :</label>
                                            <input
                                                type="date"
                                                value={testStartDate}
                                                onChange={(e) => setTestStartDate(e.target.value)}
                                                style={styles.input}
                                                required={splitMethod === 'date'}
                                            />
                                        </div>
                                        <div style={styles.fieldGroup}>
                                            <label style={styles.subLabel}>Date de fin Test (End) :</label>
                                            <input
                                                type="date"
                                                value={testEndDate}
                                                onChange={(e) => setTestEndDate(e.target.value)}
                                                style={styles.input}
                                                required={splitMethod === 'date'}
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Hyperparameters Form */}
                    <div style={styles.hyperContainer}>
                        <h4 style={styles.hyperTitle}>⚙️ Hyperparamètres Dynamiques ({modelType.toUpperCase()})</h4>
                        {renderHyperparameterInputs()}
                    </div>

                    <button
                        type="submit"
                        disabled={loading || !selectedVersionId}
                        style={loading || !selectedVersionId ? styles.buttonDisabled : styles.buttonPrimary}
                    >
                        {loading ? 'Entraînement en cours...' : '🚀 Lancer l\'entraînement du modèle'}
                    </button>
                </form>
            </div>

            {/* Training Results */}
            {trainingResult && (
                <div style={styles.card}>
                    <h3 style={styles.cardTitle}>📈 Résultats de l'Évaluation du Modèle Actuel</h3>
                    <div style={styles.metaInfo}>
                        <span>ID Modèle : <strong>{trainingResult.model_id}</strong></span>
                        <span>Algorithme : <strong>{trainingResult.model_type?.toUpperCase()}</strong></span>
                        <span>Cible : <strong>{trainingResult.target_column}</strong></span>
                    </div>

                    <div style={styles.metricsGrid}>
                        <div style={styles.metricCard}>
                            <span style={styles.metricLabel}>Score R²</span>
                            <span style={{ ...styles.metricValue, color: '#059669' }}>
                                {(trainingResult.metrics?.r2 * 100)?.toFixed(1)}%
                            </span>
                        </div>
                        <div style={styles.metricCard}>
                            <span style={styles.metricLabel}>RMSE</span>
                            <span style={styles.metricValue}>
                                {trainingResult.metrics?.rmse?.toFixed(2)}
                            </span>
                        </div>
                        <div style={styles.metricCard}>
                            <span style={styles.metricLabel}>MAE</span>
                            <span style={styles.metricValue}>
                                {trainingResult.metrics?.mae?.toFixed(2)}
                            </span>
                        </div>
                        <div style={styles.metricCard}>
                            <span style={styles.metricLabel}>MAPE</span>
                            <span style={{ ...styles.metricValue, color: '#d97706' }}>
                                {trainingResult.metrics?.mape?.toFixed(1)}%
                            </span>
                        </div>
                    </div>

                    <div style={{ marginTop: '20px' }}>
                        <h4 style={styles.exportTitle}>
                            Caractéristiques utilisées ({trainingResult.feature_names?.length || 0} colonnes) :
                        </h4>
                        <div style={styles.tagContainer}>
                            {trainingResult.feature_names?.map((feat, i) => (
                                <span key={i} style={styles.tag}>{feat}</span>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Model Registry */}
            <div style={styles.card}>
                <h3 style={styles.cardTitle}>📜 Registre des Modèles Enregistrés (Model Registry)</h3>
                <div style={{ overflowX: 'auto' }}>
                    <table style={styles.table}>
                        <thead>
                            <tr style={styles.thRow}>
                                <th style={styles.th}>ID Modèle</th>
                                <th style={styles.th}>Algorithme</th>
                                <th style={styles.th}>Type</th>
                                <th style={styles.th}>Colonne Cible</th>
                                <th style={styles.th}>Score R²</th>
                                <th style={styles.th}>RMSE</th>
                                <th style={styles.th}>Date</th>
                                <th style={styles.th}>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {modelRegistry.length === 0 ? (
                                <tr>
                                    <td colSpan="8" style={styles.tdEmpty}>Aucun modèle répertorié dans le registre.</td>
                                </tr>
                            ) : (
                                modelRegistry.map((m) => (
                                    <tr key={m.model_id} style={styles.tr}>
                                        <td style={{ ...styles.td, fontWeight: '600', color: '#2563eb' }}>{m.model_id}</td>
                                        <td style={{ ...styles.td, textTransform: 'uppercase' }}>{m.model_type}</td>
                                        <td style={styles.td}>
                                            <span style={styles.badge(m.target_type)}>
                                                {m.target_type}
                                            </span>
                                        </td>
                                        <td style={styles.td}>{m.target_column}</td>
                                        <td style={{ ...styles.td, fontWeight: '600', color: '#059669' }}>
                                            {(m.r2_score * 100)?.toFixed(1)}%
                                        </td>
                                        <td style={styles.td}>{m.rmse?.toFixed(2)}</td>
                                        <td style={{ ...styles.td, color: '#64748b' }}>{m.created_at || 'N/A'}</td>
                                        <td style={{ ...styles.td, display: 'flex', gap: '6px' }}>
                                            <button
                                                type="button"
                                                onClick={() => setSelectedHistoryModel(m)}
                                                style={styles.buttonActionSmall}
                                            >
                                                👁️
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => handleDeleteModel(m.model_id)}
                                                style={styles.buttonDeleteSmall}
                                            >
                                                🗑️
                                            </button>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* History Model Modal/Details */}
            {selectedHistoryModel && (
                <div style={styles.cardHistoryDetail}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <h3 style={{ margin: 0, fontSize: '16px', color: '#1e293b' }}>
                            🔍 Détails du modèle sélectionné : <span style={{ color: '#2563eb' }}>{selectedHistoryModel.model_id}</span>
                        </h3>
                        <button onClick={() => setSelectedHistoryModel(null)} style={styles.buttonClose}>
                            ✖ Fermer
                        </button>
                    </div>

                    <div style={styles.metaInfo}>
                        <span>Algorithme : <strong>{selectedHistoryModel.model_type?.toUpperCase()}</strong></span>
                        <span>Type : <strong>{selectedHistoryModel.target_type}</strong></span>
                        <span>Cible : <strong>{selectedHistoryModel.target_column}</strong></span>
                        <span>Date de création : <strong>{selectedHistoryModel.created_at || 'N/A'}</strong></span>
                    </div>

                    <div style={styles.metricsGrid}>
                        <div style={styles.metricCard}>
                            <span style={styles.metricLabel}>Score R²</span>
                            <span style={{ ...styles.metricValue, color: '#059669' }}>
                                {(selectedHistoryModel.r2_score * 100)?.toFixed(1)}%
                            </span>
                        </div>
                        <div style={styles.metricCard}>
                            <span style={styles.metricLabel}>RMSE</span>
                            <span style={styles.metricValue}>
                                {selectedHistoryModel.rmse?.toFixed(2)}
                            </span>
                        </div>
                        <div style={styles.metricCard}>
                            <span style={styles.metricLabel}>MAE</span>
                            <span style={styles.metricValue}>
                                {selectedHistoryModel.mae ? selectedHistoryModel.mae.toFixed(2) : 'N/A'}
                            </span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// ==================== STYLES ====================
const styles = {
    container: { padding: '24px', maxWidth: '1200px', margin: '0 auto', fontFamily: "'Inter', sans-serif", color: '#1e293b' },
    header: { marginBottom: '20px', borderBottom: '1px solid #e2e8f0', paddingBottom: '16px' },
    title: { margin: '0 0 8px 0', color: '#0f172a', fontSize: '22px', fontWeight: '700' },
    subtitle: { margin: 0, color: '#64748b', fontSize: '14px' },

    tabContainer: { display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap' },
    tabActive: { padding: '10px 18px', backgroundColor: '#2563eb', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '13px' },
    tabInactive: { padding: '10px 18px', backgroundColor: '#f1f5f9', color: '#475569', border: '1px solid #cbd5e1', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '13px' },

    card: { background: '#ffffff', borderRadius: '12px', padding: '24px', marginBottom: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', border: '1px solid #e2e8f0' },
    cardHistoryDetail: { background: '#f8fafc', borderRadius: '12px', padding: '24px', marginBottom: '24px', boxShadow: '0 2px 6px rgba(0,0,0,0.15)', border: '2px solid #2563eb' },
    cardTitle: { margin: '0 0 16px 0', fontSize: '16px', fontWeight: '600', color: '#334155', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' },
    grid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '16px',
        alignItems: 'stretch'
    },
    fieldGroup: {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        height: '100%'
    },
    inputGroupAlign: { display: 'flex', gap: '8px', alignItems: 'center', width: '100%' },
    label: { fontSize: '13px', fontWeight: '600', color: '#475569', margin: 0 },
    subLabel: {
        fontSize: '12px',
        fontWeight: '500',
        color: '#64748b',
        marginBottom: '8px',
        lineHeight: '1.35',
        minHeight: '34px',
        display: 'flex',
        alignItems: 'flex-end',
        wordBreak: 'break-word'
    },
    radioLabel: { fontSize: '13px', cursor: 'pointer', color: '#334155', display: 'flex', alignItems: 'center', gap: '4px' },
    select: {
        height: '42px',
        padding: '0 12px',
        borderRadius: '8px',
        border: '1px solid #cbd5e1',
        fontSize: '14px',
        backgroundColor: '#f8fafc',
        outline: 'none',
        width: '100%',
        boxSizing: 'border-box',
        marginTop: 'auto'
    },
    input: {
        height: '42px',
        padding: '0 12px',
        borderRadius: '8px',
        border: '1px solid #cbd5e1',
        fontSize: '14px',
        backgroundColor: '#ffffff',
        outline: 'none',
        width: '100%',
        boxSizing: 'border-box',
        marginTop: 'auto'
    },

    splitGroup: { marginBottom: '20px', marginTop: '16px', padding: '16px', backgroundColor: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column' },
    slider: { width: '100%', cursor: 'pointer' },
    dateBox: { backgroundColor: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '8px', padding: '12px' },
    dateBoxTitle: { margin: '0 0 10px 0', fontSize: '13px', fontWeight: '600', color: '#1e293b' },

    hyperContainer: {
        padding: '20px',
        backgroundColor: '#f1f5f9',
        borderRadius: '10px',
        marginBottom: '20px',
        border: '1px solid #cbd5e1'
    },
    hyperTitle: { margin: '0 0 12px 0', fontSize: '13px', fontWeight: '600', color: '#2563eb' },

    buttonPrimary: { width: '100%', padding: '12px 20px', backgroundColor: '#2563eb', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '14px' },
    buttonDisabled: { width: '100%', padding: '12px 20px', backgroundColor: '#94a3b8', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'not-allowed', fontSize: '14px' },
    buttonDelete: { height: '42px', padding: '0 14px', backgroundColor: '#dc2626', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '13px', whiteSpace: 'nowrap' },
    buttonDeleteDisabled: { height: '42px', padding: '0 14px', backgroundColor: '#e2e8f0', color: '#94a3b8', border: 'none', borderRadius: '8px', cursor: 'not-allowed', fontSize: '13px', whiteSpace: 'nowrap' },
    buttonActionSmall: { padding: '4px 8px', backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '6px', cursor: 'pointer', fontSize: '12px' },
    buttonDeleteSmall: { padding: '4px 8px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '6px', cursor: 'pointer', fontSize: '12px' },
    buttonClose: { padding: '6px 12px', backgroundColor: '#e2e8f0', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: '600' },

    errorAlert: { backgroundColor: '#fef2f2', color: '#b91c1c', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #fecaca', fontSize: '14px' },
    successAlert: { backgroundColor: '#ecfdf5', color: '#047857', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #a7f3d0', fontSize: '14px' },

    metaInfo: { display: 'flex', gap: '20px', flexWrap: 'wrap', fontSize: '13px', color: '#475569', backgroundColor: '#ffffff', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #e2e8f0' },
    metricsGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' },
    metricCard: { padding: '16px', backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0', textAlign: 'center' },
    metricLabel: { fontSize: '12px', color: '#64748b', display: 'block', marginBottom: '4px' },
    metricValue: { fontSize: '20px', fontWeight: '700', color: '#0f172a' },

    exportTitle: { margin: '0 0 8px 0', fontSize: '13px', color: '#334155' },
    tagContainer: { display: 'flex', flexWrap: 'wrap', gap: '6px', maxHeight: '120px', overflowY: 'auto', padding: '10px', backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0' },
    tag: { fontSize: '11px', padding: '4px 8px', backgroundColor: '#e2e8f0', color: '#334155', borderRadius: '4px', fontFamily: 'monospace' },

    table: { width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' },
    thRow: { backgroundColor: '#f8fafc', borderBottom: '1px solid #e2e8f0' },
    th: { padding: '10px 12px', fontWeight: '600', color: '#475569' },
    tr: { borderBottom: '1px solid #f1f5f9' },
    td: { padding: '10px 12px', color: '#334155' },
    tdEmpty: { padding: '16px', textAlign: 'center', color: '#94a3b8' },
    badge: (type) => {
        let bg = '#eff6ff';
        let color = '#2563eb';
        let border = '1px solid #bfdbfe';

        if (type === 'elec') {
            bg = '#fffbe2';
            color = '#d97706';
            border = '1px solid #fde68a';
        } else if (type === 'thermal') {
            bg = '#fef2f2';
            color = '#dc2626';
            border = '1px solid #fecaca';
        }

        return {
            padding: '2px 8px',
            borderRadius: '12px',
            fontSize: '11px',
            fontWeight: '600',
            backgroundColor: bg,
            color: color,
            border: border
        };
    }
};