import React, { useState, useEffect, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API_ML_BASE = 'http://localhost:8000/ml/training';
const MODEL_COLORS = ['#2563eb', '#059669', '#d97706', '#7c3aed', '#db2777', '#0891b2', '#4f46e5'];

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
    const [catParams, setCatParams] = useState({ iterations: 200, learning_rate: 0.05, depth: 6, random_state: 42 });
    const [rfParams, setRfParams] = useState({ n_estimators: 100, max_depth: 12, min_samples_split: 2, min_samples_leaf: 1, random_state: 42 });
    const [ridgeParams, setRidgeParams] = useState({ alpha: 1.0, solver: 'auto' });
    const [rnnParams, setRnnParams] = useState({ epochs: 20, batch_size: 32, hidden_dim: 64, time_steps: 12, learning_rate: 0.001, dropout: 0.2 });
    const [annParams, setAnnParams] = useState({ epochs: 30, batch_size: 32, hidden_units: '64,32', learning_rate: 0.001, dropout: 0.1 });
    const [gbParams, setGbParams] = useState({ n_estimators: 100, learning_rate: 0.05, max_depth: 5, random_state: 42 });
    const [svrParams, setSvrParams] = useState({ kernel: 'rbf', C: 1.0, epsilon: 0.1 });
    const [knnParams, setKnnParams] = useState({ n_neighbors: 5, weights: 'uniform' });

    // UI States
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);
    const [successMsg, setSuccessMsg] = useState(null);
    const [trainingResult, setTrainingResult] = useState(null);
    const [modelRegistry, setModelRegistry] = useState([]);
    const [selectedHistoryModel, setSelectedHistoryModel] = useState(null);

    // Feature Importance & Comparison States
    const [selectedForComparison, setSelectedForComparison] = useState([]);
    const [showComparison, setShowComparison] = useState(false);
    const [comparisonView, setComparisonView] = useState('table'); // 'table' | 'chart'

    // Helper pour récupérer l'unité selon le targetType
    const getUnitLabel = (type) => {
        if (type === 'visitor') return 'personne';
        return 'kWh'; // dùng chung cho 'elec' và 'thermal' (ec)
    };

    // 1. Charger la liste des attractions au démarrage
    useEffect(() => {
        fetchAttractions();
    }, []);

    // 2. Charger la liste des versions lors du changement d'attraction ou de targetType
    useEffect(() => {
        fetchVersions(selectedAttraction);
    }, [selectedAttraction, targetType]);

    // 3. Charger les colonnes et la plage de dates lors de la sélection d'une version
    useEffect(() => {
        if (selectedVersionId) {
            fetchVersionColumns(selectedVersionId, selectedAttraction);
            fetchVersionDateRange(selectedVersionId, selectedAttraction);
        } else {
            setAvailableColumns([]);
        }
    }, [selectedVersionId, selectedAttraction]);

    // 4. Charger le registre des modèles
    useEffect(() => {
        fetchModelRegistry();
    }, [targetType]);

    useEffect(() => {
        if (targetType === 'visitor') {
            setTargetColumn('visitor_count');
        } else if (targetType === 'elec') {
            setTargetColumn('elec'); // hoặc tên cột mặc định điện của bạn (vd: 'elec_kwh')
        } else if (targetType === 'thermal') {
            setTargetColumn('ec');   // hoặc tên cột mặc định nhiệt của bạn (vd: 'ec_kwh')
        }
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
                    const vIdUpper = vId.toUpperCase();

                    // 1. Kiểm tra tiền tố/hậu tố loại dữ liệu
                    if (!vId.includes('fs')) return false;
                    if (targetType === 'visitor' && !vId.includes('visitor')) return false;
                    if (targetType === 'elec' && !/_elec_|_elec$|^elec_/.test(vId)) return false;
                    if (targetType === 'thermal' && !/_ec_|_ec$|^ec_/.test(vId)) return false;

                    // Nếu chọn Tất cả (ALL) hoặc phiên bản gốc (V0_RAW)
                    if (currentAttr === 'ALL' || vIdUpper === 'V0_RAW') return true;

                    // 2. Tìm tất cả các mã attraction dạng H01, H02, H07... có trong tên version_id
                    const foundAttractions = vIdUpper.match(/H\d+/g) || [];

                    if (foundAttractions.length > 0) {
                        // Nếu trong version_id có chứa mã attraction, CHỈ giữ lại nếu đúng là mã currentAttr đang lọc (ví dụ H07)
                        return foundAttractions.includes(currentAttr);
                    }

                    // 3. Nếu trong version_id không có chuỗi 'Hxx', kiểm tra id_attraction từ object backend
                    const itemAttr = String(item.id_attraction || '').trim().toUpperCase();
                    return itemAttr === currentAttr;
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

                let numericCols = cols.filter((c) => !['datetime', 'id_attraction', 'attraction_id', 'ouvert'].includes(c));

                numericCols = numericCols.filter((col) => {
                    const c = col.toLowerCase();
                    if (targetType === 'visitor') {
                        return !c.includes('elec') && !c.includes('ec_') && !c.startsWith('ec');
                    }
                    if (targetType === 'elec') {
                        return !c.includes('chaleur') && !c.includes('visitor');
                    }
                    if (targetType === 'thermal') {
                        return !c.includes('elec') && !c.includes('visitor');
                    }
                    return true;
                });

                setAvailableColumns(numericCols);

                if (numericCols.length > 0) {
                    // Tự động tìm cột ưu tiên khớp với targetType hiện tại
                    let defaultTarget = '';

                    if (targetType === 'visitor') {
                        defaultTarget = numericCols.find(c => c.toLowerCase().includes('visitor')) || numericCols[0];
                    } else if (targetType === 'elec') {
                        defaultTarget = numericCols.find(c => c.toLowerCase().startsWith('elec') || c.toLowerCase().includes('elec')) || numericCols[0];
                    } else if (targetType === 'thermal') {
                        defaultTarget = numericCols.find(c => c.toLowerCase().startsWith('ec') || c.toLowerCase().includes('ec') || c.toLowerCase().includes('chaleur')) || numericCols[0];
                    }

                    setTargetColumn((prev) => {
                        // Nếu prev đã hợp lệ trong danh sách mới thì giữ nguyên, ngược lại chọn defaultTarget vừa tìm
                        return numericCols.includes(prev) ? prev : defaultTarget;
                    });
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
            setSelectedForComparison(prev => prev.filter(id => id !== modelId));
            fetchModelRegistry();
        } catch (err) {
            setErrorMsg(err.message);
        }
    };

    const handleBulkDeleteModels = async () => {
        if (selectedForComparison.length === 0) return;

        const confirmDelete = window.confirm(
            `Êtes-vous sûr de vouloir supprimer les ${selectedForComparison.length} modèles sélectionnés ?`
        );
        if (!confirmDelete) return;

        setLoading(true);
        setErrorMsg(null);
        setSuccessMsg(null);

        try {
            const deletePromises = selectedForComparison.map(modelId =>
                fetch(`${API_ML_BASE}/models/${modelId}`, { method: 'DELETE' })
            );

            await Promise.all(deletePromises);

            setSuccessMsg(`${selectedForComparison.length} modèle(s) supprimé(s) avec succès.`);

            setSelectedForComparison([]);
            setShowComparison(false);
            if (selectedHistoryModel && selectedForComparison.includes(selectedHistoryModel.model_id)) {
                setSelectedHistoryModel(null);
            }

            fetchModelRegistry();
        } catch (err) {
            setErrorMsg("Erreur lors de la suppression groupée des modèles.");
        } finally {
            setLoading(false);
        }
    };

    const toggleComparisonModel = (modelId) => {
        setSelectedForComparison(prev =>
            prev.includes(modelId) ? prev.filter(id => id !== modelId) : [...prev, modelId]
        );
    };

    const getActiveHyperparameters = () => {
        switch (modelType) {
            case 'linear_regression':
                return {};
            case 'xgboost':
                return {
                    n_estimators: Number(xgbParams.n_estimators),
                    learning_rate: Number(xgbParams.learning_rate),
                    max_depth: Number(xgbParams.max_depth),
                    subsample: Number(xgbParams.subsample),
                    colsample_bytree: Number(xgbParams.colsample_bytree),
                    random_state: Number(xgbParams.random_state)
                };
            case 'gradient_boosting':
                return {
                    n_estimators: Number(gbParams.n_estimators),
                    learning_rate: Number(gbParams.learning_rate),
                    max_depth: Number(gbParams.max_depth),
                    random_state: Number(gbParams.random_state)
                };
            case 'random_forest':
                return {
                    n_estimators: Number(rfParams.n_estimators),
                    max_depth: Number(rfParams.max_depth),
                    min_samples_split: Number(rfParams.min_samples_split),
                    min_samples_leaf: Number(rfParams.min_samples_leaf),
                    random_state: Number(rfParams.random_state)
                };
            case 'svr':
                return {
                    kernel: svrParams.kernel,
                    C: Number(svrParams.C),
                    epsilon: Number(svrParams.epsilon)
                };
            case 'knn':
                return {
                    n_neighbors: Number(knnParams.n_neighbors),
                    weights: knnParams.weights
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
            case 'ann':
                return {
                    epochs: Number(annParams.epochs),
                    batch_size: Number(annParams.batch_size),
                    hidden_units: annParams.hidden_units.split(',').map(n => Number(n.trim())),
                    learning_rate: Number(annParams.learning_rate),
                    dropout: Number(annParams.dropout)
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
            case 'linear_regression':
                return (
                    <p style={{ margin: 0, color: '#64748b', fontSize: '13px' }}>
                        ℹ️ Régression Linéaire classique sans hyperparamètres complexes.
                    </p>
                );

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

            case 'catboost':
                return (
                    <div style={styles.grid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Nombre d'itérations (iterations)</label>
                            <input type="number" value={catParams.iterations} onChange={(e) => setCatParams({ ...catParams, iterations: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Taux d'apprentissage (learning_rate)</label>
                            <input type="number" step="0.01" value={catParams.learning_rate} onChange={(e) => setCatParams({ ...catParams, learning_rate: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Profondeur d'arbre (depth)</label>
                            <input type="number" value={catParams.depth} onChange={(e) => setCatParams({ ...catParams, depth: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Graine aléatoire (random_state)</label>
                            <input type="number" value={catParams.random_state} onChange={(e) => setCatParams({ ...catParams, random_state: e.target.value })} style={styles.input} />
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
            case 'gradient_boosting':
                return (
                    <div style={styles.grid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Nombre d'arbres (n_estimators)</label>
                            <input type="number" value={gbParams.n_estimators} onChange={(e) => setGbParams({ ...gbParams, n_estimators: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Taux d'apprentissage (learning_rate)</label>
                            <input type="number" step="0.01" value={gbParams.learning_rate} onChange={(e) => setGbParams({ ...gbParams, learning_rate: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Profondeur maximale (max_depth)</label>
                            <input type="number" value={gbParams.max_depth} onChange={(e) => setGbParams({ ...gbParams, max_depth: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Graine aléatoire (random_state)</label>
                            <input type="number" value={gbParams.random_state} onChange={(e) => setGbParams({ ...gbParams, random_state: e.target.value })} style={styles.input} />
                        </div>
                    </div>
                );

            case 'svr':
                return (
                    <div style={styles.grid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Noyau (kernel)</label>
                            <select value={svrParams.kernel} onChange={(e) => setSvrParams({ ...svrParams, kernel: e.target.value })} style={styles.select}>
                                <option value="rbf">rbf</option>
                                <option value="linear">linear</option>
                                <option value="poly">poly</option>
                                <option value="sigmoid">sigmoid</option>
                            </select>
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Paramètre de pénalité (C)</label>
                            <input type="number" step="0.1" value={svrParams.C} onChange={(e) => setSvrParams({ ...svrParams, C: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Tolérance d'erreur (epsilon)</label>
                            <input type="number" step="0.01" value={svrParams.epsilon} onChange={(e) => setSvrParams({ ...svrParams, epsilon: e.target.value })} style={styles.input} />
                        </div>
                    </div>
                );

            case 'knn':
                return (
                    <div style={styles.grid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Nombre de voisins (n_neighbors)</label>
                            <input type="number" value={knnParams.n_neighbors} onChange={(e) => setKnnParams({ ...knnParams, n_neighbors: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Pondération (weights)</label>
                            <select value={knnParams.weights} onChange={(e) => setKnnParams({ ...knnParams, weights: e.target.value })} style={styles.select}>
                                <option value="uniform">uniform</option>
                                <option value="distance">distance</option>
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

            case 'ann':
                return (
                    <div style={styles.grid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Époques (epochs)</label>
                            <input type="number" value={annParams.epochs} onChange={(e) => setAnnParams({ ...annParams, epochs: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Taille du lot (batch_size)</label>
                            <input type="number" value={annParams.batch_size} onChange={(e) => setAnnParams({ ...annParams, batch_size: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Neurones par couche (ex: 64,32)</label>
                            <input type="text" value={annParams.hidden_units} onChange={(e) => setAnnParams({ ...annParams, hidden_units: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Taux d'apprentissage (learning_rate)</label>
                            <input type="number" step="0.0001" value={annParams.learning_rate} onChange={(e) => setAnnParams({ ...annParams, learning_rate: e.target.value })} style={styles.input} />
                        </div>
                        <div style={styles.fieldGroup}>
                            <label style={styles.subLabel}>Taux d'abandon (dropout)</label>
                            <input type="number" step="0.05" value={annParams.dropout} onChange={(e) => setAnnParams({ ...annParams, dropout: e.target.value })} style={styles.input} />
                        </div>
                    </div>
                );

            default:
                return null;
        }
    };

    // Style des badges par type
    const getBadgeStyle = (type) => {
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
    };

    const comparisonModelsList = modelRegistry.filter(m => selectedForComparison.includes(m.model_id));

    // Dữ liệu đã biến đổi: Trục X là 3 Metric chính, mỗi cột là một mô hình đứng cạnh nhau
    const groupedMetricChartData = useMemo(() => {
        if (!selectedForComparison.length) return [];

        const models = modelRegistry.filter(m => selectedForComparison.includes(m.model_id));
        const unit = getUnitLabel(targetType);

        const dataR2 = { metric: 'R² (%)' };
        const dataRMSE = { metric: `RMSE (${unit})` };
        const dataMAE = { metric: `MAE (${unit})` };

        models.forEach(m => {
            const modelLabel = m.model_id;
            dataR2[modelLabel] = m.r2_score ? Number((m.r2_score * 100).toFixed(1)) : 0;
            dataRMSE[modelLabel] = m.rmse ? Number(m.rmse.toFixed(2)) : 0;
            dataMAE[modelLabel] = m.mae ? Number(m.mae.toFixed(2)) : 0;
        });

        return [dataR2, dataRMSE, dataMAE];
    }, [selectedForComparison, modelRegistry, targetType]);

    return (
        <div style={styles.container}>
            {/* En-tête */}
            <div style={styles.header}>
                <h2 style={styles.title}>🧠 Studio d'Apprentissage Automatique (Entraînement des Modèles)</h2>
                <p style={styles.subtitle}>
                    Entraînement, optimisation des hyperparamètres, explicabilité (SHAP / PFI) et analyse automatique par IA.
                </p>
            </div>

            {/* Sélecteur de type de cible */}
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

            {/* Formulaire Principal */}
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
                                <optgroup label="Machine Learning">
                                    <option value="xgboost">XGBoost Regressor</option>
                                    <option value="gradient_boosting">Gradient Boosting Regressor</option>
                                    <option value="random_forest">Random Forest Regressor</option>
                                    <option value="svr">Support Vector Regressor (SVR)</option>
                                    <option value="knn">K-Nearest Neighbors (KNN)</option>
                                    <option value="ridge">Ridge Regression</option>
                                    <option value="linear_regression">Linear Regression</option>
                                </optgroup>
                                <optgroup label="Deep Learning">
                                    <option value="lstm">LSTM (Long Short-Term Memory)</option>
                                    <option value="gru">GRU (Gated Recurrent Unit)</option>
                                    <option value="ann">ANN (Artificial Neural Network)</option>
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

                    {/* Formulaire des Hyperparamètres */}
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

            {/* Résultats de l'Entraînement Actuel */}
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
                            <span style={styles.metricLabel}>RMSE ({getUnitLabel(targetType)})</span>
                            <span style={styles.metricValue}>
                                {trainingResult.metrics?.rmse?.toFixed(2)}
                            </span>
                        </div>
                        <div style={styles.metricCard}>
                            <span style={styles.metricLabel}>MAE ({getUnitLabel(targetType)})</span>
                            <span style={styles.metricValue}>
                                {trainingResult.metrics?.mae?.toFixed(2)}
                            </span>
                        </div>
                    </div>

                    {/* 🤖 Explication par IA (Gemini AI Analyst) */}
                    {trainingResult.ai_explanation && (
                        <div style={styles.aiBox}>
                            <h4 style={styles.aiTitle}>🤖 Explication de l'Analyste IA (Gemini) :</h4>
                            <p style={styles.aiText}>
                                {trainingResult.ai_explanation}
                            </p>
                        </div>
                    )}

                    {/* 📊 Section Explicabilité : SHAP vs Permutation Importance (PFI) */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginTop: '20px' }}>
                        {/* SHAP Feature Importance */}
                        {trainingResult.shap_importance && trainingResult.shap_importance.length > 0 && (
                            <div style={styles.importanceBox}>
                                <h4 style={styles.exportTitle}>🔷 Importance SHAP (Top Caractéristiques) :</h4>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '10px' }}>
                                    {trainingResult.shap_importance.slice(0, 8).map((item) => (
                                        <div key={item.feature} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                            <span style={{ width: '150px', fontSize: '12px', color: '#334155', fontFamily: 'monospace' }}>{item.feature}</span>
                                            <div style={styles.barBackground}>
                                                <div style={{ width: `${Math.min(item.importance * 100, 100)}%`, backgroundColor: '#2563eb', height: '100%' }}></div>
                                            </div>
                                            <span style={{ width: '50px', fontSize: '11px', fontWeight: '600', color: '#1e293b' }}>
                                                {item.importance.toFixed(3)}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Permutation Feature Importance (PFI) */}
                        {trainingResult.pfi_importance && trainingResult.pfi_importance.length > 0 && (
                            <div style={styles.importanceBox}>
                                <h4 style={styles.exportTitle}>🔶 Permutation Importance - PFI (Top Caractéristiques) :</h4>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '10px' }}>
                                    {trainingResult.pfi_importance.slice(0, 8).map((item) => (
                                        <div key={item.feature} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                            <span style={{ width: '150px', fontSize: '12px', color: '#334155', fontFamily: 'monospace' }}>{item.feature}</span>
                                            <div style={styles.barBackground}>
                                                <div style={{ width: `${Math.min(item.importance * 100, 100)}%`, backgroundColor: '#d97706', height: '100%' }}></div>
                                            </div>
                                            <span style={{ width: '50px', fontSize: '11px', fontWeight: '600', color: '#1e293b' }}>
                                                {item.importance.toFixed(3)}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
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

            {/* Vue Comparatives des Modèles */}
            {showComparison && comparisonModelsList.length > 0 && (
                <div style={styles.cardHistoryDetail}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <h3 style={{ margin: 0, fontSize: '16px', color: '#1e293b' }}>
                            ⚖️ Comparaison des Modèles Sélectionnés ({comparisonModelsList.length})
                        </h3>
                        <div style={{ display: 'flex', gap: '8px' }}>
                            <button
                                onClick={() => setComparisonView('table')}
                                style={comparisonView === 'table' ? styles.toggleActive : styles.toggleInactive}
                            >
                                📋 Tableau
                            </button>
                            <button
                                onClick={() => setComparisonView('chart')}
                                style={comparisonView === 'chart' ? styles.toggleActive : styles.toggleInactive}
                            >
                                📊 Graphique en barres
                            </button>
                            <button onClick={() => setShowComparison(false)} style={styles.buttonClose}>
                                ✖ Fermer
                            </button>
                        </div>
                    </div>

                    {comparisonView === 'table' ? (
                        <div style={{ overflowX: 'auto' }}>
                            <table style={styles.table}>
                                <thead>
                                    <tr style={styles.thRow}>
                                        <th style={styles.th}>ID Modèle</th>
                                        <th style={styles.th}>Algorithme</th>
                                        <th style={styles.th}>Cible</th>
                                        <th style={styles.th}>Score R²</th>
                                        <th style={styles.th}>RMSE ({getUnitLabel(targetType)})</th>
                                        <th style={styles.th}>MAE ({getUnitLabel(targetType)})</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {comparisonModelsList.map((m) => (
                                        <tr key={m.model_id} style={styles.tr}>
                                            <td style={{ ...styles.td, fontWeight: '600', color: '#2563eb' }}>{m.model_id}</td>
                                            <td style={{ ...styles.td, textTransform: 'uppercase' }}>{m.model_type}</td>
                                            <td style={styles.td}>{m.target_column}</td>
                                            <td style={{ ...styles.td, fontWeight: '700', color: '#059669' }}>
                                                {(m.r2_score * 100)?.toFixed(1)}%
                                            </td>
                                            <td style={styles.td}>{m.rmse?.toFixed(2)}</td>
                                            <td style={styles.td}>{m.mae ? m.mae.toFixed(2) : 'N/A'}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <div style={{ width: '100%', height: 400, marginTop: '16px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={groupedMetricChartData}
                                    margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                                    barGap={6} // Khoảng cách giữa các mô hình trong cùng 1 metric
                                    barCategoryGap="25%" // Khoảng cách giữa các nhóm R², RMSE, MAE
                                >
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                    <XAxis dataKey="metric" stroke="#64748b" fontSize={13} fontWeight="bold" tickLine={false} />
                                    <YAxis stroke="#64748b" fontSize={12} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #cbd5e1', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                                    />
                                    <Legend wrapperStyle={{ paddingTop: '10px' }} />

                                    {/* Mỗi mô hình là 1 cột có màu riêng xếp sát cạnh nhau theo từng nhóm Metric */}
                                    {selectedForComparison.map((modelId, index) => {
                                        const color = MODEL_COLORS[index % MODEL_COLORS.length];
                                        return (
                                            <Bar
                                                key={modelId}
                                                dataKey={modelId}
                                                name={modelId}
                                                fill={color}
                                                radius={[4, 4, 0, 0]}
                                                maxBarSize={45}
                                            />
                                        );
                                    })}
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </div>
            )}

            {/* Registre des Modèles (Model Registry) */}
            <div style={styles.card}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
                    <h3 style={{ ...styles.cardTitle, marginBottom: 0, borderBottom: 'none' }}>
                        📜 Registre des Modèles Enregistrés (Model Registry)
                    </h3>

                    {selectedForComparison.length > 0 && (
                        <div style={{ display: 'flex', gap: '10px' }}>
                            <button
                                onClick={() => setShowComparison(true)}
                                style={{ ...styles.buttonPrimary, width: 'auto', padding: '6px 14px', fontSize: '13px' }}
                            >
                                ⚖️ Comparer ({selectedForComparison.length}) Modèles
                            </button>
                            <button
                                onClick={handleBulkDeleteModels}
                                disabled={loading}
                                style={{ ...styles.buttonDelete, height: 'auto', padding: '6px 14px', fontSize: '13px' }}
                            >
                                🗑️ Supprimer ({selectedForComparison.length})
                            </button>
                        </div>
                    )}
                </div>

                <div style={{ overflowX: 'auto' }}>
                    <table style={styles.table}>
                        <thead>
                            <tr style={styles.thRow}>
                                <th style={styles.th}>Comparer</th>
                                <th style={styles.th}>ID Modèle</th>
                                <th style={styles.th}>Algorithme</th>
                                <th style={styles.th}>Type</th>
                                <th style={styles.th}>Colonne Cible</th>
                                <th style={styles.th}>Score R²</th>
                                <th style={styles.th}>RMSE ({getUnitLabel(targetType)})</th>
                                <th style={styles.th}>Date</th>
                                <th style={styles.th}>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {modelRegistry.length === 0 ? (
                                <tr>
                                    <td colSpan="9" style={styles.tdEmpty}>Aucun modèle répertorié dans le registre.</td>
                                </tr>
                            ) : (
                                modelRegistry.map((m) => (
                                    <tr key={m.model_id} style={styles.tr}>
                                        <td style={{ ...styles.td, textAlign: 'center' }}>
                                            <input
                                                type="checkbox"
                                                checked={selectedForComparison.includes(m.model_id)}
                                                onChange={() => toggleComparisonModel(m.model_id)}
                                            />
                                        </td>
                                        <td style={{ ...styles.td, fontWeight: '600', color: '#2563eb' }}>{m.model_id}</td>
                                        <td style={{ ...styles.td, textTransform: 'uppercase' }}>{m.model_type}</td>
                                        <td style={styles.td}>
                                            <span style={getBadgeStyle(m.target_type)}>
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

            {/* Modal / Détails d'un modèle de l'historique */}
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
                            <span style={styles.metricLabel}>RMSE ({getUnitLabel(selectedHistoryModel.target_type || targetType)})</span>
                            <span style={styles.metricValue}>
                                {selectedHistoryModel.rmse?.toFixed(2)}
                            </span>
                        </div>
                        <div style={styles.metricCard}>
                            <span style={styles.metricLabel}>MAE ({getUnitLabel(selectedHistoryModel.target_type || targetType)})</span>
                            <span style={styles.metricValue}>
                                {selectedHistoryModel.mae ? selectedHistoryModel.mae.toFixed(2) : 'N/A'}
                            </span>
                        </div>
                    </div>

                    {/* Explication IA sauvegardée */}
                    {selectedHistoryModel.ai_explanation && (
                        <div style={{ ...styles.aiBox, marginTop: '16px' }}>
                            <h4 style={styles.aiTitle}>💡 Explication de l'Analyste IA :</h4>
                            <p style={styles.aiText}>
                                {selectedHistoryModel.ai_explanation}
                            </p>
                        </div>
                    )}
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

    toggleActive: { padding: '6px 12px', backgroundColor: '#2563eb', color: '#ffffff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: '600' },
    toggleInactive: { padding: '6px 12px', backgroundColor: '#e2e8f0', color: '#475569', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: '600' },

    errorAlert: { backgroundColor: '#fef2f2', color: '#b91c1c', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #fecaca', fontSize: '14px' },
    successAlert: { backgroundColor: '#ecfdf5', color: '#047857', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #a7f3d0', fontSize: '14px' },

    metaInfo: { display: 'flex', gap: '20px', flexWrap: 'wrap', fontSize: '13px', color: '#475569', backgroundColor: '#ffffff', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #e2e8f0' },
    metricsGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' },
    metricCard: { padding: '16px', backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0', textAlign: 'center' },
    metricLabel: { fontSize: '12px', color: '#64748b', display: 'block', marginBottom: '4px' },
    metricValue: { fontSize: '20px', fontWeight: '700', color: '#0f172a' },

    aiBox: { marginTop: '20px', padding: '16px', backgroundColor: '#f0f9ff', borderLeft: '4px solid #0284c7', borderRadius: '6px' },
    aiTitle: { margin: '0 0 8px 0', color: '#0369a1', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '6px' },
    aiText: { margin: 0, fontSize: '13px', color: '#334155', lineHeight: '1.6', whiteSpace: 'pre-line' },

    importanceBox: { background: '#f8fafc', padding: '16px', borderRadius: '8px', border: '1px solid #e2e8f0' },
    barBackground: { flex: 1, backgroundColor: '#e2e8f0', borderRadius: '4px', height: '8px', overflow: 'hidden' },

    exportTitle: { margin: '0 0 8px 0', fontSize: '13px', color: '#334155' },
    tagContainer: { display: 'flex', flexWrap: 'wrap', gap: '6px', maxHeight: '120px', overflowY: 'auto', padding: '10px', backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0' },
    tag: { fontSize: '11px', padding: '4px 8px', backgroundColor: '#e2e8f0', color: '#334155', borderRadius: '4px', fontFamily: 'monospace' },

    table: { width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' },
    thRow: { backgroundColor: '#f8fafc', borderBottom: '1px solid #e2e8f0' },
    th: { padding: '10px 12px', fontWeight: '600', color: '#475569' },
    tr: { borderBottom: '1px solid #f1f5f9' },
    td: { padding: '10px 12px', color: '#334155' },
    tdEmpty: { padding: '16px', textAlign: 'center', color: '#94a3b8' }
};