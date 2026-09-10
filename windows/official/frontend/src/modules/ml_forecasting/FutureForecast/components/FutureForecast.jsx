import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
    ResponsiveContainer,
    AreaChart,
    Area,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    Cell,
    Legend
} from 'recharts';

const API_BASE_URL = 'http://localhost:8000/predict-visitor';

export default function FutureForecast() {
    const [activeTab, setActiveTab] = useState('simulation');

    const [availableModels, setAvailableModels] = useState([]);
    const [availableAttractions, setAvailableAttractions] = useState([]);
    const [loadingModels, setLoadingModels] = useState(false);

    // ==================== TAB 1: SIMULATION SCÉNARIO STATES ====================
    const [simTargetType, setSimTargetType] = useState('visitor');
    const [simSelectedModelId, setSimSelectedModelId] = useState('');
    const [simVisitorModelId, setSimVisitorModelId] = useState('');
    const [simAttraction, setSimAttraction] = useState('ATT_01');
    const [simStartDate, setSimStartDate] = useState('2026-09-10T00:00');
    const [simEndDate, setSimEndDate] = useState('2026-09-10T23:00');
    const [simFreqType, setSimFreqType] = useState(1);
    const [simTempOffset, setSimTempOffset] = useState(0.0);
    const [simOpFactor, setSimOpFactor] = useState(1.0);

    // CÁC TRẠNG THÁI CHO OUVERT_MODE
    const [simOuvertMode, setSimOuvertMode] = useState('ideal'); // 'ideal' hoặc 'profile'
    const [simOpenThreshold, setSimOpenThreshold] = useState(0.1);

    // Trạng thái cho bảng preview, danh sách cột động
    const [previewFeatures, setPreviewFeatures] = useState([]);
    const [previewColumns, setPreviewColumns] = useState([]);
    const [rawRequiredCols, setRawRequiredCols] = useState([]);
    const [showAllColumns, setShowAllColumns] = useState(false);
    const [loadingPreview, setLoadingPreview] = useState(false);

    const [simResults, setSimResults] = useState(null);
    const [simLoading, setSimLoading] = useState(false);

    // ==================== TAB 2: DIRECT INFERENCE STATES ====================
    const [selectedModelId, setSelectedModelId] = useState('');
    const [modelMetadata, setModelMetadata] = useState(null);
    const [featureInputs, setFeatureInputs] = useState({});
    const [predictionResult, setPredictionResult] = useState(null);
    const [featureImportances, setFeatureImportances] = useState([]);

    const [loadingFeatures, setLoadingFeatures] = useState(false);
    const [predicting, setPredicting] = useState(false);

    const [errorMsg, setErrorMsg] = useState(null);
    const [successMsg, setSuccessMsg] = useState(null);

    useEffect(() => {
        fetchTrainedModels();
        fetchAvailableAttractions();
    }, []);

    // Tự động gán model đầu tiên tương ứng với target khi danh sách model hoặc target_type thay đổi
    useEffect(() => {
        if (availableModels.length > 0) {
            const visitorModel = availableModels.find(m => m.target === 'visitor');
            if (visitorModel) {
                setSimVisitorModelId(visitorModel.model_id);
            }

            const matchedModels = availableModels.filter(m => {
                if (simTargetType === 'visitor') return m.target === 'visitor';
                if (simTargetType === 'electricity') return m.target === 'electricity' || m.target === 'elec';
                if (simTargetType === 'thermal') return m.target === 'thermal';
                return true;
            });

            if (matchedModels.length > 0) {
                setSimSelectedModelId(matchedModels[0].model_id);
            } else {
                setSimSelectedModelId(availableModels[0].model_id);
            }

            if (!selectedModelId) setSelectedModelId(availableModels[0].model_id);
        }
    }, [availableModels, simTargetType]);

    useEffect(() => {
        if (selectedModelId && activeTab === 'direct_inference') {
            fetchModelFeatures(selectedModelId);
        } else {
            setModelMetadata(null);
            setFeatureInputs({});
        }
    }, [selectedModelId, activeTab]);

    // Cập nhật lại danh sách cột hiển thị khi chuyển đổi giữa `showAllColumns`
    useEffect(() => {
        if (previewFeatures.length > 0) {
            if (rawRequiredCols.length > 0) {
                const finalCols = rawRequiredCols.includes('datetime')
                    ? rawRequiredCols
                    : ['datetime', ...rawRequiredCols];
                setPreviewColumns(finalCols);
            } else {
                setPreviewColumns(Object.keys(previewFeatures[0]));
            }
        }
    }, [previewFeatures, rawRequiredCols]);

    const fetchTrainedModels = async () => {
        setLoadingModels(true);
        setErrorMsg(null);
        try {
            const res = await axios.get(`${API_BASE_URL}/ml/models`);
            const models = res.data || [];
            setAvailableModels(models);
        } catch (err) {
            setErrorMsg("Impossible de charger la liste des modèles entraînés.");
        } finally {
            setLoadingModels(false);
        }
    };

    const fetchAvailableAttractions = async () => {
        try {
            const res = await axios.get(`${API_BASE_URL}/attractions`);
            if (res.data && res.data.attractions && res.data.attractions.length > 0) {
                setAvailableAttractions(res.data.attractions);
                setSimAttraction(res.data.attractions[0]);
            }
        } catch (err) {
            setAvailableAttractions(['ATT_01', 'ATT_02', 'ATT_03', 'ATT_AERO_01']);
        }
    };

    const formatLocalISO = (dateTimeStr) => {
        if (!dateTimeStr) return '';
        return dateTimeStr.length === 16 ? `${dateTimeStr}:00` : dateTimeStr;
    };

    // Gọi API lấy danh sách tất cả các biến đầu vào theo giờ của Mô hình đã chọn
    const handleFetchPreview = async () => {
        setLoadingPreview(true);
        setErrorMsg(null);
        try {
            const payload = {
                target_type: simTargetType,
                model_id: simSelectedModelId || undefined,
                id_attraction: simAttraction,
                start_date: formatLocalISO(simStartDate),
                end_date: formatLocalISO(simEndDate),
                type_frequentation_default: Number(simFreqType),
                ouvert_mode: simOuvertMode,
                open_threshold: Number(simOpenThreshold),
                temp_offset: Number(simTempOffset),
                operation_factor: Number(simOpFactor)
            };
            const res = await axios.post(`${API_BASE_URL}/forecast/preview-features`, payload);
            const featuresData = res.data.features || [];
            const requiredCols = res.data.model_required_features || [];

            setPreviewFeatures(featuresData);
            setRawRequiredCols(requiredCols);

            setSuccessMsg("📋 Variables d'entrée générées avec succès !");
        } catch (err) {
            setErrorMsg("Erreur lors de la génération des variables d'entrée.");
        } finally {
            setLoadingPreview(false);
        }
    };

    const handleFeatureCellChange = (index, fieldName, val) => {
        const updated = [...previewFeatures];
        if (val === '') {
            updated[index][fieldName] = '';
        } else {
            const parsedNum = Number(val);
            updated[index][fieldName] = isNaN(parsedNum) ? val : parsedNum;
        }
        setPreviewFeatures(updated);
    };

    const handleRunSimulation = async (e) => {
        e.preventDefault();
        setSimLoading(true);
        setErrorMsg(null);
        setSuccessMsg(null);

        try {
            const payload = {
                target_type: simTargetType,
                model_id: simSelectedModelId || undefined,
                visitor_model_id: simTargetType !== 'visitor' ? (simVisitorModelId || undefined) : undefined,
                id_attraction: simAttraction,
                start_date: formatLocalISO(simStartDate),
                end_date: formatLocalISO(simEndDate),
                type_frequentation_default: Number(simFreqType),
                ouvert_mode: simOuvertMode,
                open_threshold: Number(simOpenThreshold),
                temp_offset: Number(simTempOffset),
                operation_factor: Number(simOpFactor),
                custom_hourly_features: previewFeatures.length > 0 ? previewFeatures : undefined
            };

            const res = await axios.post(`${API_BASE_URL}/forecast`, payload);
            setSimResults(res.data);
            setSuccessMsg("🎉 Simulation par heure calculée avec succès !");
        } catch (err) {
            console.error(err);
            setErrorMsg(err.response?.data?.detail || "Erreur lors du calcul de la simulation.");
        } finally {
            setSimLoading(false);
        }
    };

    const visitorModels = availableModels.filter(m => m.target === 'visitor');
    const targetModels = availableModels.filter(m => {
        if (simTargetType === 'visitor') return m.target === 'visitor';
        if (simTargetType === 'electricity') return m.target === 'electricity' || m.target === 'elec';
        if (simTargetType === 'thermal') return m.target === 'thermal';
        return true;
    });

    const fetchModelFeatures = async (modelId) => {
        setLoadingFeatures(true);
        setErrorMsg(null);
        setPredictionResult(null);

        try {
            const res = await axios.get(`${API_BASE_URL}/ml/models/${modelId}`);
            const meta = res.data;
            setModelMetadata(meta);

            const initialInputs = {};
            if (meta.features && Array.isArray(meta.features)) {
                meta.features.forEach(feat => {
                    const featName = typeof feat === 'object' ? feat.name : feat;
                    const defaultVal = typeof feat === 'object' && feat.default_value !== undefined ? feat.default_value : 0;
                    initialInputs[featName] = defaultVal;
                });
            }
            if (initialInputs['id_attraction'] !== undefined && simAttraction) {
                initialInputs['id_attraction'] = simAttraction;
            }

            setFeatureInputs(initialInputs);
            setFeatureImportances(meta.feature_importances || []);
        } catch (err) {
            setErrorMsg("Erreur de récupération des paramètres du modèle sélectionné.");
        } finally {
            setLoadingFeatures(false);
        }
    };

    const handleFeatureInputChange = (featName, value) => {
        setFeatureInputs(prev => ({
            ...prev,
            [featName]: value
        }));
    };

    const handleRunPrediction = async (e) => {
        e.preventDefault();
        setPredicting(true);
        setErrorMsg(null);
        setSuccessMsg(null);

        try {
            const payload = {
                model_id: selectedModelId,
                features: featureInputs
            };

            const res = await axios.post(`${API_BASE_URL}/ml/predict`, payload);
            setPredictionResult(res.data);
            setSuccessMsg("🎉 Inférence exécutée avec succès !");
        } catch (err) {
            console.error(err);
            setErrorMsg(err.response?.data?.detail || "Erreur lors de l'exécution de la prédiction.");
        } finally {
            setPredicting(false);
        }
    };

    return (
        <div style={styles.container}>
            <div style={styles.header}>
                <h2 style={styles.title}>🚀 Future Forecast Engine (Hourly Chained Pipeline)</h2>
                <p style={styles.subtitle}>
                    Moteur d'IA prédictive horaire : Chaînage dynamique des prédictions de fréquentation vers la consommation d'énergie.
                </p>
            </div>

            <div style={styles.tabContainer}>
                <button
                    style={activeTab === 'simulation' ? styles.tabActive : styles.tabInactive}
                    onClick={() => { setActiveTab('simulation'); setErrorMsg(null); setSuccessMsg(null); }}
                >
                    📈 1. Simulation Temporelle Par Heure (Chaînage IA)
                </button>
                <button
                    style={activeTab === 'direct_inference' ? styles.tabActive : styles.tabInactive}
                    onClick={() => { setActiveTab('direct_inference'); setErrorMsg(null); setSuccessMsg(null); }}
                >
                    🤖 2. Inférence Directe (Instant)
                </button>
            </div>

            {errorMsg && <div style={styles.errorAlert}>⚠️ {errorMsg}</div>}
            {successMsg && <div style={styles.successAlert}>✅ {successMsg}</div>}

            {activeTab === 'simulation' && (
                <div>
                    <form onSubmit={handleRunSimulation}>
                        <div style={styles.card}>
                            <h3 style={styles.cardTitle}>⚙️ Configuration du Scénario Prédictif Horaire</h3>

                            <div style={styles.configGrid}>
                                <div style={styles.fieldGroup}>
                                    <label style={styles.label}>Cible Finale de Prédiction</label>
                                    <select style={styles.select} value={simTargetType} onChange={(e) => setSimTargetType(e.target.value)}>
                                        <option value="visitor">👥 Fréquentation Visiteurs (pers.)</option>
                                        <option value="electricity">⚡ Consommation Électrique (kWh)</option>
                                        <option value="thermal">🔥 Énergie Thermique (kWh)</option>
                                    </select>
                                </div>

                                {simTargetType !== 'visitor' && (
                                    <div style={styles.fieldGroup}>
                                        <label style={styles.label}>1.Modèle IA Visiteurs</label>
                                        <select
                                            style={styles.select}
                                            value={simVisitorModelId}
                                            onChange={(e) => setSimVisitorModelId(e.target.value)}
                                            disabled={loadingModels}
                                        >
                                            {visitorModels.map(m => (
                                                <option key={m.model_id} value={m.model_id}>{m.name || m.model_id}</option>
                                            ))}
                                        </select>
                                    </div>
                                )}

                                <div style={styles.fieldGroup}>
                                    <label style={styles.label}>
                                        {simTargetType !== 'visitor' ? "2. Modèle IA Cible Énergie" : "Modèle IA Visiteurs"}
                                    </label>
                                    <select
                                        style={styles.select}
                                        value={simSelectedModelId}
                                        onChange={(e) => setSimSelectedModelId(e.target.value)}
                                        disabled={loadingModels}
                                    >
                                        {targetModels.map(m => (
                                            <option key={m.model_id} value={m.model_id}>{m.name || m.model_id}</option>
                                        ))}
                                    </select>
                                </div>

                                <div style={styles.fieldGroup}>
                                    <label style={styles.label}>Identifiant Attraction</label>
                                    <select
                                        style={styles.select}
                                        value={simAttraction}
                                        onChange={(e) => setSimAttraction(e.target.value)}
                                    >
                                        {availableAttractions.map((attr, idx) => (
                                            <option key={idx} value={attr}>{attr}</option>
                                        ))}
                                    </select>
                                </div>

                                <div style={styles.fieldGroup}>
                                    <label style={styles.label}>Début</label>
                                    <input style={styles.input} type="datetime-local" value={simStartDate} onChange={(e) => setSimStartDate(e.target.value)} required />
                                </div>

                                <div style={styles.fieldGroup}>
                                    <label style={styles.label}>Fin</label>
                                    <input style={styles.input} type="datetime-local" value={simEndDate} onChange={(e) => setSimEndDate(e.target.value)} required />
                                </div>

                                {/* <div style={styles.fieldGroup}>
                                    <label style={styles.label}>Type de Journée</label>
                                    <select style={styles.select} value={simFreqType} onChange={(e) => setSimFreqType(e.target.value)}>
                                        <option value={1}>Jour Normal</option>
                                        <option value={2}>Week-end</option>
                                        <option value={3}>Jour Férié</option>
                                    </select>
                                </div> */}

                                <div style={styles.fieldGroup}>
                                    <label style={styles.label}>Mode Ouverture (Ouvert)</label>
                                    <select style={styles.select} value={simOuvertMode} onChange={(e) => setSimOuvertMode(e.target.value)}>
                                        <option value="ideal">🌟 Idéal (1.0 si profil &gt; seuil)</option>
                                        <option value="profile">📊 Profil Moyen Historique (0.0 - 1.0)</option>
                                    </select>
                                </div>

                                {/* {simOuvertMode === 'ideal' && (
                                    <div style={styles.fieldGroup}>
                                        <label style={styles.label}>Seuil d'Ouverture (Threshold)</label>
                                        <input
                                            style={styles.input}
                                            type="number"
                                            step="0.05"
                                            min="0"
                                            max="1"
                                            value={simOpenThreshold}
                                            onChange={(e) => setSimOpenThreshold(e.target.value)}
                                        />
                                    </div>
                                )} */}

                                {/* <div style={styles.fieldGroup}>
                                    <label style={styles.label}>Écart Météo (°C)</label>
                                    <input style={styles.input} type="number" step="0.5" value={simTempOffset} onChange={(e) => setSimTempOffset(e.target.value)} />
                                </div>

                                <div style={styles.fieldGroup}>
                                    <label style={styles.label}>Facteur d'Exploitation (0 - 1)</label>
                                    <input style={styles.input} type="number" step="0.05" min="0" max="1" value={simOpFactor} onChange={(e) => setSimOpFactor(e.target.value)} />
                                </div> */}
                            </div>

                            <div style={{ marginTop: '20px', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                                <button type="button" onClick={handleFetchPreview} disabled={loadingPreview} style={styles.buttonSecondary}>
                                    {loadingPreview ? '⏳ Génération des features...' : '🔍 Charger & Inspecter les Variables d\'Entrée de ce Modèle'}
                                </button>
                            </div>

                            {/* Bảng hiển thị và chỉnh sửa các biến đầu vào */}
                            {previewFeatures.length > 0 && (
                                <div style={{ marginTop: '20px', background: '#f8fafc', padding: '16px', borderRadius: '8px', border: '1px solid #cbd5e1' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                                        <h4 style={{ margin: 0, fontSize: '15px', color: '#1e293b' }}>
                                            ✏️ Tableau de Contrôle des Variables ({previewFeatures.length} heures, {previewColumns.length} colonnes)
                                        </h4>
                                    </div>
                                    <div style={{ maxHeight: '350px', overflowX: 'auto', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '6px' }}>
                                        <table style={{ width: '100%', borderCollapse: 'collapse', backgroundColor: '#fff', fontSize: '13px', whiteSpace: 'nowrap' }}>
                                            <thead>
                                                <tr style={{ backgroundColor: '#f1f5f9', borderBottom: '1px solid #cbd5e1', textAlign: 'left' }}>
                                                    {previewColumns.map((colName) => (
                                                        <th key={colName} style={{ padding: '8px 12px', fontWeight: '600', color: '#334155' }}>
                                                            {colName}
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {previewFeatures.map((row, idx) => (
                                                    <tr key={idx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                        {previewColumns.map((colName) => (
                                                            <td key={colName} style={{ padding: '6px 8px' }}>
                                                                <input
                                                                    type={typeof row[colName] === 'number' ? 'number' : 'text'}
                                                                    step="any"
                                                                    value={row[colName] ?? ''}
                                                                    onChange={(e) => handleFeatureCellChange(idx, colName, e.target.value)}
                                                                    style={styles.tableInput}
                                                                    readOnly={colName === 'datetime'}
                                                                />
                                                            </td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            )}

                            <div style={{ marginTop: '24px' }}>
                                <button type="submit" disabled={simLoading} style={simLoading ? styles.buttonDisabled : styles.buttonPrimary}>
                                    {simLoading ? '⏳ Calcul de la chaîne de prédiction horaire...' : '⚡ Lancer la Simulation Chaînée par Heure'}
                                </button>
                            </div>
                        </div>
                    </form>

                    {simResults && (() => {
                        // Tự động tính toán các chỉ số thống kê nếu Backend trả về kết quả thành công
                        const preds = simResults.predictions || [];
                        const values = preds.map(p => Number(p.predicted_value) || 0);
                        const visitors = preds.map(p => Number(p.visitor_count) || 0);

                        const calcStats = (arr) => {
                            if (arr.length === 0) return { total: 0, mean: 0, median: 0, max: 0, min: 0 };
                            const sorted = [...arr].sort((a, b) => a - b);
                            const total = arr.reduce((a, b) => a + b, 0);
                            const mean = total / arr.length;
                            const mid = Math.floor(sorted.length / 2);
                            const median = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
                            return {
                                total: Math.round(total * 100) / 100,
                                mean: Math.round(mean * 100) / 100,
                                median: Math.round(median * 100) / 100,
                                max: Math.round(Math.max(...arr) * 100) / 100,
                                min: Math.round(Math.min(...arr) * 100) / 100
                            };
                        };

                        const predStats = simResults.summary?.predicted_value || calcStats(values);
                        const visitorStats = simResults.summary?.visitor_count || (visitors.some(v => v > 0) ? calcStats(visitors) : null);
                        const unit = simResults.summary?.unit || (simResults.target_type === 'visitor' ? 'pers.' : 'kWh');

                        return (
                            <div style={styles.card}>
                                <h3 style={styles.cardTitle}>
                                    📊 Résultats de la Simulation Horaire ({simResults.total_records || preds.length} heures)
                                </h3>
                                <div style={{ fontSize: '13px', color: '#64748b', marginBottom: '16px' }}>
                                    🔹 Modèle Cible: <code>{simResults.target_model_used || simSelectedModelId}</code> <br />
                                    {simResults.visitor_model_used && (
                                        <span>🔹 Modèle Visitor Chaîné: <code>{simResults.visitor_model_used}</code></span>
                                    )}
                                </div>

                                {/* 🟢 KHỐI KPI & BẢNG TỔNG HỢP (SUMMARY TABLE) */}
                                <div style={{ marginBottom: '24px' }}>
                                    <h4 style={{ fontSize: '14px', fontWeight: '700', color: '#334155', marginBottom: '12px' }}>
                                        📌 Tableau Récapitulatif des Prédictions ({unit})
                                    </h4>

                                    {/* 1. KHỐI THẺ KPI */}
                                    <div style={styles.kpiGrid}>
                                        <div style={{ ...styles.kpiCard, borderLeft: '4px solid #2563eb' }}>
                                            <span style={styles.kpiTitle}>Total Prédit ({simResults.target_type})</span>
                                            <span style={{ ...styles.kpiValue, color: '#2563eb' }}>
                                                {predStats.total.toLocaleString()} <small style={{ fontSize: '13px' }}>{unit}</small>
                                            </span>
                                            <span style={styles.kpiSub}>Somme sur {preds.length}h</span>
                                        </div>

                                        <div style={{ ...styles.kpiCard, borderLeft: '4px solid #0284c7' }}>
                                            <span style={styles.kpiTitle}>Moyenne Horaire</span>
                                            <span style={{ ...styles.kpiValue, color: '#0284c7' }}>
                                                {predStats.mean.toLocaleString()} <small style={{ fontSize: '13px' }}>{unit}</small>
                                            </span>
                                            <span style={styles.kpiSub}>Valeur moyenne par heure</span>
                                        </div>

                                        {visitorStats && (
                                            <div style={{ ...styles.kpiCard, borderLeft: '4px solid #10b981' }}>
                                                <span style={styles.kpiTitle}>Total Visiteurs (Chaîné)</span>
                                                <span style={{ ...styles.kpiValue, color: '#10b981' }}>
                                                    {visitorStats.total.toLocaleString()} <small style={{ fontSize: '13px' }}>pers.</small>
                                                </span>
                                                <span style={styles.kpiSub}>Fréquentation estimée</span>
                                            </div>
                                        )}
                                    </div>

                                    {/* 2. BẢNG THỐNG KÊ CHI TIẾT */}
                                    <div style={{ marginTop: '16px', overflowX: 'auto', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
                                        <table style={{ width: '100%', borderCollapse: 'collapse', backgroundColor: '#ffffff', fontSize: '13px', textAlign: 'center' }}>
                                            <thead>
                                                <tr style={{ backgroundColor: '#f8fafc', borderBottom: '1px solid #cbd5e1', color: '#475569' }}>
                                                    <th style={{ padding: '10px', textAlign: 'left' }}>Métrique</th>
                                                    <th style={{ padding: '10px' }}>Total (Somme)</th>
                                                    <th style={{ padding: '10px' }}>Moyenne (Mean)</th>
                                                    <th style={{ padding: '10px' }}>Médiane (Median)</th>
                                                    <th style={{ padding: '10px' }}>Maximum (Max)</th>
                                                    <th style={{ padding: '10px' }}>Minimum (Min)</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                    <td style={{ padding: '10px', textAlign: 'left', fontWeight: '600', color: '#2563eb' }}>
                                                        {(simResults.target_type || 'VALUE').toUpperCase()} ({unit})
                                                    </td>
                                                    <td style={{ padding: '10px', fontWeight: '700' }}>{predStats.total.toLocaleString()}</td>
                                                    <td style={{ padding: '10px' }}>{predStats.mean.toLocaleString()}</td>
                                                    <td style={{ padding: '10px' }}>{predStats.median.toLocaleString()}</td>
                                                    <td style={{ padding: '10px', color: '#dc2626', fontWeight: '600' }}>{predStats.max.toLocaleString()}</td>
                                                    <td style={{ padding: '10px', color: '#16a34a', fontWeight: '600' }}>{predStats.min.toLocaleString()}</td>
                                                </tr>

                                                {/* THAY ĐỔI Ở ĐÂY: Thêm điều kiện simResults.target_type !== 'visitor' */}
                                                {visitorStats && simResults.target_type !== 'visitor' && (
                                                    <tr>
                                                        <td style={{ padding: '10px', textAlign: 'left', fontWeight: '600', color: '#10b981' }}>
                                                            Visiteurs (pers.)
                                                        </td>
                                                        <td style={{ padding: '10px', fontWeight: '700' }}>{visitorStats.total.toLocaleString()}</td>
                                                        <td style={{ padding: '10px' }}>{visitorStats.mean.toLocaleString()}</td>
                                                        <td style={{ padding: '10px' }}>{visitorStats.median.toLocaleString()}</td>
                                                        <td style={{ padding: '10px', color: '#dc2626', fontWeight: '600' }}>{visitorStats.max.toLocaleString()}</td>
                                                        <td style={{ padding: '10px', color: '#16a34a', fontWeight: '600' }}>{visitorStats.min.toLocaleString()}</td>
                                                    </tr>
                                                )}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>

                                {/* BIỂU ĐỒ DÒNG THỜI GIAN */}
                                <div style={{ width: '100%', height: 380, marginTop: '16px' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={preds} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                                            <defs>
                                                <linearGradient id="colorPred" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.8} />
                                                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                                                </linearGradient>
                                                <linearGradient id="colorVis" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                                                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <XAxis dataKey="heure" tickFormatter={(h) => `${h}h`} />
                                            <YAxis yAxisId="left" orientation="left" />
                                            <YAxis yAxisId="right" orientation="right" />
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <Tooltip />
                                            <Legend />
                                            <Area yAxisId="left" type="monotone" dataKey="predicted_value" name={`Prédiction ${simResults.target_type}`} stroke="#2563eb" fillOpacity={1} fill="url(#colorPred)" />
                                            {simResults.target_type !== 'visitor' && (
                                                <Area yAxisId="right" type="monotone" dataKey="visitor_count" name="Visiteurs Prédits (Chaîné)" stroke="#10b981" fillOpacity={0.3} fill="url(#colorVis)" />
                                            )}
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        );
                    })()}
                </div>
            )}

            {activeTab === 'direct_inference' && (
                <div>
                    <div style={styles.card}>
                        <h3 style={styles.cardTitle}>1. Sélectionner un Modèle IA Registré</h3>
                        <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
                            <div style={{ flex: 1, minWidth: '280px' }}>
                                <select style={styles.select} value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)} disabled={loadingModels}>
                                    {availableModels.map((m) => (
                                        <option key={m.model_id} value={m.model_id}>{m.name || m.model_id} [Cible: {m.target}]</option>
                                    ))}
                                </select>
                            </div>
                            {modelMetadata && (
                                <div style={styles.badgeInfo}>
                                    <span>Algorithme : <strong>{modelMetadata.algorithm}</strong></span>
                                    <span>Variable Cible : <strong>{modelMetadata.target_variable}</strong></span>
                                </div>
                            )}
                        </div>
                    </div>

                    {loadingFeatures ? (
                        <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                            ⏳ Chargement des métadonnées du modèle...
                        </div>
                    ) : modelMetadata && (
                        <form onSubmit={handleRunPrediction}>
                            <div style={styles.card}>
                                <h3 style={styles.cardTitle}>
                                    2. Saisie des Variables d'Entrée Instantanée ({Object.keys(featureInputs).length} features)
                                </h3>

                                <div style={styles.configGrid}>
                                    {modelMetadata.features && modelMetadata.features.map((feat, idx) => {
                                        const featName = typeof feat === 'object' ? feat.name : feat;
                                        const featLabel = typeof feat === 'object' ? (feat.label || feat.name) : feat;

                                        if (featName === 'id_attraction') {
                                            return (
                                                <div key={idx} style={styles.fieldGroup}>
                                                    <label style={styles.label}>{featLabel} <span style={{ color: '#ef4444' }}>*</span></label>
                                                    <select
                                                        style={styles.select}
                                                        value={featureInputs[featName] || availableAttractions[0]}
                                                        onChange={(e) => handleFeatureInputChange(featName, e.target.value)}
                                                    >
                                                        {availableAttractions.map((attr, aIdx) => (
                                                            <option key={aIdx} value={attr}>{attr}</option>
                                                        ))}
                                                    </select>
                                                    <span style={styles.fieldHint}>Variable: <code>{featName}</code></span>
                                                </div>
                                            );
                                        }

                                        return (
                                            <div key={idx} style={styles.fieldGroup}>
                                                <label style={styles.label}>{featLabel} <span style={{ color: '#ef4444' }}>*</span></label>
                                                <input
                                                    type="number"
                                                    step="any"
                                                    style={styles.input}
                                                    value={featureInputs[featName] !== undefined ? featureInputs[featName] : ''}
                                                    onChange={(e) => handleFeatureInputChange(featName, e.target.value === '' ? '' : Number(e.target.value))}
                                                    required
                                                />
                                                <span style={styles.fieldHint}>Variable: <code>{featName}</code></span>
                                            </div>
                                        );
                                    })}
                                </div>

                                <div style={{ marginTop: '24px' }}>
                                    <button type="submit" disabled={predicting} style={predicting ? styles.buttonDisabled : styles.buttonPrimary}>
                                        {predicting ? '⏳ Inférence en cours...' : '⚡ Calculer l\'Inférence Instantanée'}
                                    </button>
                                </div>
                            </div>
                        </form>
                    )}

                    {predictionResult && (
                        <div style={{ ...styles.card, borderLeft: '6px solid #2563eb' }}>
                            <h3 style={styles.cardTitle}>🎯 Résultat de l'Inférence Directe</h3>
                            <div style={styles.kpiGrid}>
                                <div style={styles.kpiCard}>
                                    <span style={styles.kpiTitle}>Valeur Prédite</span>
                                    <span style={{ ...styles.kpiValue, color: '#2563eb' }}>
                                        {predictionResult.predicted_value.toLocaleString()} <small style={{ fontSize: '14px' }}>{predictionResult.unit}</small>
                                    </span>
                                    <span style={styles.kpiSub}>Modèle: {predictionResult.model_id}</span>
                                </div>

                                {predictionResult.confidence_interval && (
                                    <div style={styles.kpiCard}>
                                        <span style={styles.kpiTitle}>Intervalle de Confiance (95%)</span>
                                        <span style={{ ...styles.kpiValue, color: '#059669', fontSize: '18px', marginTop: '6px' }}>
                                            [{predictionResult.confidence_interval[0]} — {predictionResult.confidence_interval[1]}]
                                        </span>
                                    </div>
                                )}
                            </div>

                            {featureImportances.length > 0 && (
                                <div style={{ marginTop: '24px' }}>
                                    <h4 style={{ fontSize: '14px', color: '#475569', marginBottom: '12px' }}>
                                        📊 Importance des Variables (Feature Importances)
                                    </h4>
                                    <div style={{ width: '100%', height: 220 }}>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={featureImportances} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
                                                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                                <XAxis type="number" />
                                                <YAxis dataKey="name" type="category" fontSize={12} />
                                                <Tooltip />
                                                <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                                                    {featureImportances.map((entry, index) => (
                                                        <Cell key={`cell-${index}`} fill={index === 0 ? '#2563eb' : '#93c5fd'} />
                                                    ))}
                                                </Bar>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

const styles = {
    container: { padding: '24px', maxWidth: '1200px', margin: '0 auto', fontFamily: "'Inter', sans-serif", color: '#1e293b' },
    header: { marginBottom: '20px', borderBottom: '1px solid #e2e8f0', paddingBottom: '16px' },
    title: { margin: '0 0 8px 0', color: '#0f172a', fontSize: '22px', fontWeight: '700' },
    subtitle: { margin: 0, color: '#64748b', fontSize: '14px' },
    tabContainer: { display: 'flex', gap: '8px', marginBottom: '24px', borderBottom: '2px solid #e2e8f0' },
    tabActive: { padding: '12px 20px', border: 'none', borderBottom: '3px solid #2563eb', background: 'transparent', color: '#2563eb', fontWeight: '700', fontSize: '14px', cursor: 'pointer' },
    tabInactive: { padding: '12px 20px', border: 'none', background: 'transparent', color: '#64748b', fontWeight: '500', fontSize: '14px', cursor: 'pointer' },
    card: { background: '#ffffff', borderRadius: '12px', padding: '24px', marginBottom: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.08)', border: '1px solid #e2e8f0' },
    cardTitle: { margin: '0 0 16px 0', fontSize: '16px', fontWeight: '600', color: '#334155', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' },
    configGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' },
    fieldGroup: { display: 'flex', flexDirection: 'column', gap: '6px' },
    label: { fontSize: '13px', fontWeight: '600', color: '#475569' },
    fieldHint: { fontSize: '11px', color: '#94a3b8' },
    select: { height: '42px', padding: '0 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '14px', backgroundColor: '#f8fafc', width: '100%' },
    input: { height: '40px', padding: '0 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '14px', backgroundColor: '#ffffff', width: '100%', boxSizing: 'border-box' },
    badgeInfo: { display: 'flex', gap: '16px', padding: '10px 16px', backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '8px', fontSize: '13px', color: '#1e40af' },
    buttonPrimary: { width: '100%', padding: '12px 20px', backgroundColor: '#2563eb', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '14px' },
    buttonSecondary: { flex: 1, padding: '12px 20px', backgroundColor: '#475569', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '14px' },
    buttonToggle: { padding: '12px 20px', backgroundColor: '#0284c7', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '14px' },
    buttonDisabled: { width: '100%', padding: '12px 20px', backgroundColor: '#94a3b8', color: '#ffffff', border: 'none', borderRadius: '8px', cursor: 'not-allowed', fontSize: '14px' },
    tableInput: { width: '110px', minWidth: '90px', padding: '4px 6px', borderRadius: '4px', border: '1px solid #cbd5e1', fontSize: '13px' },
    errorAlert: { backgroundColor: '#fef2f2', color: '#b91c1c', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #fecaca', fontSize: '14px' },
    successAlert: { backgroundColor: '#ecfdf5', color: '#047857', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #a7f3d0', fontSize: '14px' },
    kpiGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginTop: '12px' },
    kpiCard: { background: '#f8fafc', border: '1px solid #e2e8f0', padding: '16px', borderRadius: '10px', display: 'flex', flexDirection: 'column', gap: '4px' },
    kpiTitle: { fontSize: '12px', color: '#64748b', fontWeight: '600', textTransform: 'uppercase' },
    kpiValue: { fontSize: '24px', fontWeight: '800', color: '#0f172a' },
    kpiSub: { fontSize: '12px', color: '#64748b' }
};