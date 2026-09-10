import React, { useState, useEffect, useMemo } from 'react';
import * as XLSX from 'xlsx';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    Brush
} from 'recharts';

const API_TRAINING_BASE = 'http://localhost:8000/ml/training';
const API_INFERENCE_BASE = 'http://localhost:8000/ml/inference';

export default function MLInference() {
    // ==================== ÉTATS (STATES) ====================
    const [attractions, setAttractions] = useState([]);
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');

    const [versions, setVersions] = useState([]);
    const [allAttractionVersions, setAllAttractionVersions] = useState([]);
    const [selectedVersionId, setSelectedVersionId] = useState('');

    const [targetType, setTargetType] = useState('visitor');
    const [modelRegistry, setModelRegistry] = useState([]);
    const [selectedModelId, setSelectedModelId] = useState('');
    const [selectedCompareModels, setSelectedCompareModels] = useState([]);

    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');

    const [inferenceResult, setInferenceResult] = useState(null);
    const [predictionList, setPredictionList] = useState([]);
    const [htmlDashboard, setHtmlDashboard] = useState('');

    const [activeViewTab, setActiveViewTab] = useState('dashboard');
    const [actionTab, setActionTab] = useState('impute'); // 'impute', 'export_db', 'export_file'

    const [targetColumn, setTargetColumn] = useState('visitor_count');
    const [imputationMode, setImputationMode] = useState('fill_missing');
    const [filterMissingOnly, setFilterMissingOnly] = useState(false);

    const [targetSchema, setTargetSchema] = useState('predictions');
    const [outputTableName, setOutputTableName] = useState('');
    const [ifExistsMode, setIfExistsMode] = useState('replace');

    const [loading, setLoading] = useState(false);
    const [loadingDashboardHtml, setLoadingDashboardHtml] = useState(false);
    const [applyingDb, setApplyingDb] = useState(false);
    const [exportingSchema, setExportingSchema] = useState(false);
    const [exportingFile, setExportingFile] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);
    const [successMsg, setSuccessMsg] = useState(null);

    const [showActual, setShowActual] = useState(true);
    const [showPredicted, setShowPredicted] = useState(true);

    const getUnitLabel = (type) => {
        if (type === 'visitor') return 'visiteurs';
        return 'kWh';
    };

    // Format Datetime : YYYY-MM-DD HH:mm:ss.SSS
    const formatDateTimeStandard = (dateStr) => {
        if (!dateStr) return '';
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return dateStr;

        const pad = (n, width = 2) => String(n).padStart(width, '0');

        const year = d.getFullYear();
        const month = pad(d.getMonth() + 1);
        const day = pad(d.getDate());
        const hours = pad(d.getHours());
        const minutes = pad(d.getMinutes());
        const seconds = pad(d.getSeconds());
        const ms = pad(d.getMilliseconds(), 3);

        return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}.${ms}`;
    };

    const formatXAxisDate = (tickItem) => {
        if (!tickItem) return '';
        const d = new Date(tickItem);
        if (isNaN(d.getTime())) return tickItem;

        const day = String(d.getDate()).padStart(2, '0');
        const month = String(d.getMonth() + 1).padStart(2, '0');
        return `${day}/${month}`;
    };

    // ==================== LỌC MÔ HÌNH TƯƠNG ĐỒNG ====================
    // Lọc danh sách các mô hình có cùng thuộc tính mục tiêu (target_column / target_type) với mô hình đang chọn
    const comparableModels = useMemo(() => {
        if (!selectedModelId || modelRegistry.length === 0) return modelRegistry;

        const currentModel = modelRegistry.find(m => m.model_id === selectedModelId);
        if (!currentModel) return modelRegistry;

        return modelRegistry.filter(m => {
            if (currentModel.target_column && m.target_column) {
                return m.target_column === currentModel.target_column;
            }
            if (currentModel.target_type && m.target_type) {
                return m.target_type === currentModel.target_type;
            }
            return true;
        });
    }, [modelRegistry, selectedModelId]);

    // Tự động làm sạch selectedCompareModels khi mô hình chính đổi
    useEffect(() => {
        if (selectedModelId) {
            const validCompareIds = selectedCompareModels.filter(id => 
                comparableModels.some(m => m.model_id === id)
            );
            
            if (validCompareIds.length === 0) {
                setSelectedCompareModels([selectedModelId]);
            } else {
                setSelectedCompareModels(validCompareIds);
            }
        }
    }, [selectedModelId, comparableModels]);

    // Hàm tự động tìm và đồng bộ Table Source theo Modèle được chọn
    const syncVersionWithModel = (modelObj, currentVersions) => {
        if (!modelObj) return;

        const availableVersions = currentVersions || versions;
        if (!availableVersions || availableVersions.length === 0) return;

        // 1. Nếu trong đối tượng model có khai báo trực tiếp tên version/table
        const directVersion = modelObj.version_id || modelObj.table_name || modelObj.dataset_version;
        if (directVersion && availableVersions.includes(directVersion)) {
            setSelectedVersionId(directVersion);
            return;
        }

        // 2. Tìm kiếm thông minh theo từ khóa trong tên mô hình
        const modelIdUpper = String(modelObj.model_id || '').toUpperCase();

        const matchingVer = availableVersions.find(v => {
            const vUpper = v.toUpperCase();
            return modelIdUpper.includes(vUpper) || vUpper.includes(modelIdUpper);
        });

        if (matchingVer) {
            setSelectedVersionId(matchingVer);
        } else {
            setSelectedVersionId(availableVersions[0]);
        }
    };

    useEffect(() => {
        fetchAttractions();
    }, []);

    useEffect(() => {
        fetchVersions(selectedAttraction);
        fetchModelRegistry(selectedAttraction, targetType);
    }, [selectedAttraction, targetType]);

    useEffect(() => {
        if (selectedVersionId) {
            fetchVersionDateRange(selectedVersionId, selectedAttraction);
            if (selectedModelId) {
                const cleanVer = selectedVersionId.replace(/-/g, '_').toLowerCase();
                const cleanMod = selectedModelId.replace(/-/g, '_').toLowerCase();
                setOutputTableName(`${cleanVer}_pred_${cleanMod}`);
            }
        }
    }, [selectedVersionId, selectedAttraction, selectedModelId]);

    // ==================== API BACKEND ====================

    const fetchAttractions = async () => {
        try {
            const res = await fetch(`${API_TRAINING_BASE}/attractions`);
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

    const fetchVersions = async (attractionId, currentTargetType) => {
        try {
            const currentAttr = String(attractionId || 'ALL').trim().toUpperCase();
            const activeTab = currentTargetType || targetType;
            const params = new URLSearchParams({ id_attraction: currentAttr });

            const res = await fetch(`${API_TRAINING_BASE}/versions?${params.toString()}`);

            if (res.ok) {
                const data = await res.json();
                let rawList = Array.isArray(data.versions) ? data.versions : (Array.isArray(data) ? data : []);

                let tableNames = rawList.map(item => {
                    if (typeof item === 'string') return item;
                    if (typeof item === 'object' && item !== null) {
                        return item.version_id || item.table_name || item.id || '';
                    }
                    return String(item);
                }).filter(name => name !== '' && !name.toLowerCase().includes('registry'));

                // 1. Filtrer les tables pour l'attraction sélectionnée
                let filteredByAttr = tableNames;
                if (currentAttr !== 'ALL') {
                    filteredByAttr = tableNames.filter(tableName => {
                        const upperName = tableName.toUpperCase();
                        const attrRegex = new RegExp(`(_|^)${currentAttr}(_|$)`, 'i');
                        return attrRegex.test(upperName);
                    });
                }
                setAllAttractionVersions(filteredByAttr);

                // 2. Tables sources filtrées par type de cible (feature store)
                const sourceFilteredList = filteredByAttr.filter(tableName => {
                    const lowerName = tableName.toLowerCase();

                    if (activeTab === 'visitor') {
                        return lowerName.includes('visitor') && lowerName.includes('fs');
                    }

                    if (activeTab === 'elec') {
                        const isElec = lowerName.includes('elec') || lowerName.includes('electricity');
                        const isThermal = lowerName.includes('thermal') || lowerName.includes('chaleur') || lowerName.includes('eau_chaude');
                        return isElec && !isThermal;
                    }

                    if (activeTab === 'thermal') {
                        const isThermal = lowerName.includes('thermal') || lowerName.includes('chaleur') || lowerName.includes('eau_chaude') || lowerName.includes('_ec_') || lowerName.endsWith('_ec');
                        const isElec = lowerName.includes('elec') || lowerName.includes('electricity');
                        return isThermal && !isElec;
                    }

                    return true;
                });

                setVersions(sourceFilteredList);

                if (selectedModelId && modelRegistry.length > 0) {
                    const currentModel = modelRegistry.find(m => m.model_id === selectedModelId);
                    if (currentModel) {
                        syncVersionWithModel(currentModel, sourceFilteredList);
                        return;
                    }
                }

                if (sourceFilteredList.length > 0) {
                    setSelectedVersionId(sourceFilteredList[0]);
                } else if (filteredByAttr.length > 0) {
                    setSelectedVersionId(filteredByAttr[0]);
                } else {
                    setSelectedVersionId('');
                }
            } else {
                setVersions([]);
                setAllAttractionVersions([]);
                setSelectedVersionId('');
            }
        } catch (e) {
            console.error('Erreur lors du chargement des versions:', e);
            setVersions([]);
            setAllAttractionVersions([]);
            setSelectedVersionId('');
        }
    };

    const fetchModelRegistry = async (attractionId, target) => {
        try {
            const res = await fetch(`${API_TRAINING_BASE}/models?target_type=${target}`);
            if (res.ok) {
                const data = await res.json();
                const list = data.models || [];
                const currentAttr = String(attractionId || 'ALL').trim().toUpperCase();

                const filteredModels = list.filter((m) => {
                    if (currentAttr === 'ALL') return true;
                    const mAttr = String(m.id_attraction || m.attraction_id || '').trim().toUpperCase();
                    const mName = String(m.model_id || '').trim().toUpperCase();
                    return mAttr === currentAttr || mName.includes(`_${currentAttr}`);
                });

                setModelRegistry(filteredModels);
                if (filteredModels.length > 0) {
                    const firstModel = filteredModels[0];
                    setSelectedModelId(firstModel.model_id);
                    setSelectedCompareModels([firstModel.model_id]);
                    if (firstModel.target_column) setTargetColumn(firstModel.target_column);

                    syncVersionWithModel(firstModel, versions);
                } else {
                    setSelectedModelId('');
                    setSelectedCompareModels([]);
                }
            }
        } catch (e) {
            console.error('Erreur lors du chargement du registre de modèles:', e);
            setModelRegistry([]);
            setSelectedModelId('');
        }
    };

    const fetchVersionDateRange = async (versionId, attractionId) => {
        if (!versionId) return;
        try {
            const params = new URLSearchParams({ id_attraction: attractionId });
            const res = await fetch(`${API_TRAINING_BASE}/versions/${versionId}/date-range?${params.toString()}`);
            if (res.ok) {
                const data = await res.json();
                if (data.min_date && data.max_date) {
                    setStartDate(data.min_date.split('T')[0]);
                    setEndDate(data.max_date.split('T')[0]);
                }
            }
        } catch (e) {
            console.error('Erreur lors du chargement de la plage de dates:', e);
        }
    };

    // ==================== GESTIONNAIRES D'ÉVÉNEMENTS ====================

    const handleRunInference = async (e) => {
        if (e) e.preventDefault();
        if (!selectedModelId || !selectedVersionId) {
            setErrorMsg('Veuillez sélectionner un modèle et une version de données valides.');
            return;
        }

        setLoading(true);
        setErrorMsg(null);
        setSuccessMsg(null);
        setInferenceResult(null);

        const payload = {
            model_id: selectedModelId,
            version_id: selectedVersionId,
            id_attraction: selectedAttraction,
            start_date: startDate || null,
            end_date: endDate || null
        };

        try {
            const res = await fetch(`${API_INFERENCE_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Erreur lors de l’exécution de la prédiction.');

            let filteredPredictions = data.predictions || [];
            if (selectedAttraction !== 'ALL') {
                filteredPredictions = filteredPredictions.filter(item => {
                    const itemAttr = String(item.id_attraction || item.attraction_id || selectedAttraction).trim().toUpperCase();
                    return itemAttr === String(selectedAttraction).trim().toUpperCase();
                });
            }

            setInferenceResult(data);
            setPredictionList(filteredPredictions);
            if (data.target_column) setTargetColumn(data.target_column);
            setSuccessMsg(`Inférence réussie pour ${selectedAttraction} ! ${filteredPredictions.length} points calculés.`);

            fetchDashboardHtml([selectedModelId]);
        } catch (err) {
            setErrorMsg(err.message);
        } finally {
            setLoading(false);
        }
    };

    const fetchDashboardHtml = async (modelsToCompare) => {
        if (!selectedVersionId || modelsToCompare.length === 0) return;

        setLoadingDashboardHtml(true);
        try {
            const payload = {
                version_id: selectedVersionId,
                model_ids: modelsToCompare,
                id_attraction: selectedAttraction,
                start_date: startDate || null,
                end_date: endDate || null
            };

            const res = await fetch(`${API_INFERENCE_BASE}/dashboard-html`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Impossible de générer le graphique Plotly.');
            }

            const htmlText = await res.text();
            setHtmlDashboard(htmlText);
        } catch (err) {
            console.error('Erreur lors du chargement du graphique Plotly:', err);
        } finally {
            setLoadingDashboardHtml(false);
        }
    };

    const handleApplyImputation = async () => {
        if (!predictionList || predictionList.length === 0) return;

        const confirmApply = window.confirm(
            `Mettre à jour la table '${selectedVersionId}' pour '${selectedAttraction}' avec le mode [${imputationMode.toUpperCase()}] ?`
        );
        if (!confirmApply) return;

        setApplyingDb(true);
        setErrorMsg(null);
        setSuccessMsg(null);

        const imputationData = predictionList.map(item => ({
            datetime: item.datetime,
            predicted_value: Number(item.predicted)
        }));

        const payload = {
            version_id: selectedVersionId,
            id_attraction: selectedAttraction,
            target_column: targetColumn,
            mode: imputationMode,
            data: imputationData
        };

        try {
            const res = await fetch(`${API_INFERENCE_BASE}/apply-imputation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Échec de la mise à jour de la base de données.');

            setSuccessMsg(data.message || `${data.updated_count} enregistrements mis à jour avec succès.`);
        } catch (err) {
            setErrorMsg(err.message);
        } finally {
            setApplyingDb(false);
        }
    };

    const handleExportToSchema = async () => {
        if (!selectedModelId || !selectedVersionId) return;

        const confirmExport = window.confirm(
            `Exporter la table '${selectedVersionId}' avec les prédictions vers le schéma '${targetSchema}' ?`
        );
        if (!confirmExport) return;

        setExportingSchema(true);
        setErrorMsg(null);
        setSuccessMsg(null);

        const payload = {
            model_id: selectedModelId,
            version_id: selectedVersionId,
            id_attraction: selectedAttraction,
            start_date: startDate || null,
            end_date: endDate || null,
            target_schema: targetSchema,
            output_table_name: outputTableName.trim() || null,
            if_exists: ifExistsMode
        };

        try {
            const res = await fetch(`${API_INFERENCE_BASE}/export-prediction-schema`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Échec de l’exportation.');

            setSuccessMsg(
                `🎉 Table exportée avec succès dans ${data.target_schema}."${data.target_table}" (${data.total_rows} lignes)`
            );
        } catch (err) {
            setErrorMsg(err.message);
        } finally {
            setExportingSchema(false);
        }
    };

    const handleExportFile = async (fileFormat) => {
        if (!selectedVersionId) {
            setErrorMsg('Veuillez sélectionner une version de données valide.');
            return;
        }

        const modelsToExport = selectedCompareModels.length > 0
            ? selectedCompareModels
            : (selectedModelId ? [selectedModelId] : []);

        if (modelsToExport.length === 0) {
            setErrorMsg('Veuillez sélectionner au moins un modèle à exporter.');
            return;
        }

        setExportingFile(true);
        setErrorMsg(null);
        setSuccessMsg(null);

        const payload = {
            version_id: selectedVersionId,
            model_ids: modelsToExport,
            file_format: fileFormat,
            id_attraction: selectedAttraction,
            start_date: startDate || null,
            end_date: endDate || null
        };

        try {
            const res = await fetch(`${API_INFERENCE_BASE}/export-file`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Échec de l’exportation du fichier.');
            }

            const contentDisposition = res.headers.get('Content-Disposition');
            let filename = `predictions_${selectedVersionId}.${fileFormat}`;
            if (contentDisposition) {
                const match = contentDisposition.match(/filename="?([^"]+)"?/);
                if (match && match[1]) {
                    filename = match[1];
                }
            }

            const blob = await res.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.setAttribute('download', filename);
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(downloadUrl);

            setSuccessMsg(`📂 Exportation réussie : ${filename}`);
        } catch (err) {
            setErrorMsg(err.message);
        } finally {
            setExportingFile(false);
        }
    };

    const handleValueChange = (index, newValue) => {
        const updated = [...predictionList];
        updated[index].predicted = parseFloat(newValue) || 0;
        setPredictionList(updated);
    };

    const handleToggleCompareModel = (mId) => {
        let updated;
        if (selectedCompareModels.includes(mId)) {
            updated = selectedCompareModels.filter(id => id !== mId);
        } else {
            updated = [...selectedCompareModels, mId];
        }
        setSelectedCompareModels(updated);
        if (updated.length > 0) {
            fetchDashboardHtml(updated);
        }
    };

    const displayedPredictions = useMemo(() => {
        let list = predictionList;
        if (selectedAttraction !== 'ALL') {
            list = list.filter(item => {
                const itemAttr = String(item.id_attraction || item.attraction_id || selectedAttraction).trim().toUpperCase();
                return itemAttr === String(selectedAttraction).trim().toUpperCase();
            });
        }
        if (!filterMissingOnly) return list;
        return list.filter(p => p.actual === null || p.actual === undefined || isNaN(p.actual));
    }, [predictionList, filterMissingOnly, selectedAttraction]);

    // ==================== CALCUL DES INDICATEURS (KPIs) ====================
    const dashboardMetrics = useMemo(() => {
        if (!predictionList || predictionList.length === 0) return null;

        let totalPoints = predictionList.length;
        let missingCount = 0;
        let validPairsCount = 0;

        let sumActual = 0;
        let sumPredicted = 0;
        let sumAbsoluteError = 0;
        let sumSquaredError = 0;
        let sumPercentageError = 0;

        const residualsDistribution = [];

        predictionList.forEach((item) => {
            const pred = Number(item.predicted) || 0;
            sumPredicted += pred;

            const isActNull = item.actual === null || item.actual === undefined || isNaN(item.actual);
            if (isActNull) {
                missingCount++;
            } else {
                const act = Number(item.actual);
                validPairsCount++;
                sumActual += act;

                const diff = pred - act;
                const absError = Math.abs(diff);
                sumAbsoluteError += absError;
                sumSquaredError += diff * diff;

                if (act !== 0) {
                    sumPercentageError += Math.abs(diff / act);
                }

                residualsDistribution.push({
                    datetime: item.datetime,
                    residual: parseFloat(diff.toFixed(2)),
                    absError: parseFloat(absError.toFixed(2))
                });
            }
        });

        const mae = validPairsCount > 0 ? sumAbsoluteError / validPairsCount : 0;
        const rmse = validPairsCount > 0 ? Math.sqrt(sumSquaredError / validPairsCount) : 0;
        const mape = validPairsCount > 0 ? (sumPercentageError / validPairsCount) * 100 : 0;
        const avgActual = validPairsCount > 0 ? sumActual / validPairsCount : 0;
        const avgPredicted = totalPoints > 0 ? sumPredicted / totalPoints : 0;

        let r2Score = 0;
        if (validPairsCount > 1 && avgActual !== 0) {
            let ssTot = 0;
            let ssRes = 0;
            predictionList.forEach((item) => {
                if (item.actual !== null && item.actual !== undefined && !isNaN(item.actual)) {
                    const act = Number(item.actual);
                    const pred = Number(item.predicted) || 0;
                    ssTot += Math.pow(act - avgActual, 2);
                    ssRes += Math.pow(act - pred, 2);
                }
            });
            r2Score = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
        }

        return {
            totalPoints,
            missingCount,
            validPairsCount,
            completionRate: parseFloat(((totalPoints - missingCount) / totalPoints * 100).toFixed(1)),
            avgActual: parseFloat(avgActual.toFixed(2)),
            avgPredicted: parseFloat(avgPredicted.toFixed(2)),
            totalPredictedSum: parseFloat(sumPredicted.toFixed(0)),
            mae: parseFloat(mae.toFixed(2)),
            rmse: parseFloat(rmse.toFixed(2)),
            mape: parseFloat(mape.toFixed(2)),
            r2Score: parseFloat((r2Score * 100).toFixed(1)),
            residualsDistribution: residualsDistribution.slice(0, 50)
        };
    }, [predictionList]);

    return (
        <div style={styles.container}>
            <div style={styles.header}>
                <h2 style={styles.title}>🔮 Studio de Prédiction & Évaluation des Modèles ML</h2>
                <p style={styles.subtitle}>
                    Analyse des données, prédiction en temps réel, comparaison des modèles et gestion PostgreSQL.
                </p>
            </div>

            <div style={styles.tabContainer}>
                <button
                    onClick={() => setTargetType('visitor')}
                    style={targetType === 'visitor' ? styles.tabActive : styles.tabInactive}
                >
                    👥 Fréquentation (Visiteurs)
                </button>
                <button
                    onClick={() => setTargetType('elec')}
                    style={targetType === 'elec' ? styles.tabActive : styles.tabInactive}
                >
                    ⚡ Consommation Électrique (kWh)
                </button>
                <button
                    onClick={() => setTargetType('thermal')}
                    style={targetType === 'thermal' ? styles.tabActive : styles.tabInactive}
                >
                    🔥 Énergie Thermique (kWh)
                </button>
            </div>

            {errorMsg && <div style={styles.errorAlert}>⚠️ {errorMsg}</div>}
            {successMsg && <div style={styles.successAlert}>✅ {successMsg}</div>}

            <div style={styles.card}>
                <h3 style={styles.cardTitle}>⚙️ Configuration par Site / Attraction</h3>
                <form onSubmit={handleRunInference}>
                    <div style={styles.configGrid}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.label}>1. Site / Attraction (Filtre Global)</label>
                            <select
                                value={selectedAttraction}
                                onChange={(e) => setSelectedAttraction(e.target.value)}
                                style={{ ...styles.select, borderColor: '#2563eb', backgroundColor: '#eff6ff' }}
                            >
                                <option value="ALL">Tous les sites (ALL)</option>
                                {attractions.map((att) => (
                                    <option key={att} value={att}>{att}</option>
                                ))}
                            </select>
                        </div>

                        <div style={styles.fieldGroup}>
                            <label style={styles.label}>2. Modèle Principal ({selectedAttraction})</label>
                            <select
                                value={selectedModelId}
                                onChange={(e) => {
                                    const modelId = e.target.value;
                                    setSelectedModelId(modelId);
                                    const m = modelRegistry.find(item => item.model_id === modelId);
                                    if (m) {
                                        if (m.target_column) setTargetColumn(m.target_column);
                                        syncVersionWithModel(m, versions);
                                    }
                                }}
                                style={styles.select}
                            >
                                {modelRegistry.length > 0 ? (
                                    modelRegistry.map((m) => (
                                        <option key={m.model_id} value={m.model_id}>
                                            🏆 {m.model_id} (R²: {m.r2_score !== undefined ? (m.r2_score * 100).toFixed(1) : 'N/A'}%)
                                        </option>
                                    ))
                                ) : (
                                    <option value="">Aucun modèle pour {selectedAttraction}</option>
                                )}
                            </select>
                        </div>

                        <div style={styles.fieldGroup}>
                            <label style={styles.label}>3. Table Source ({selectedAttraction})</label>
                            <select
                                value={selectedVersionId}
                                onChange={(e) => setSelectedVersionId(e.target.value)}
                                style={styles.select}
                            >
                                {versions.length > 0 ? (
                                    versions.map((verName) => (
                                        <option key={verName} value={verName}>
                                            {verName}
                                        </option>
                                    ))
                                ) : (
                                    <option value="">Aucune table disponible pour {selectedAttraction}</option>
                                )}
                            </select>
                        </div>

                        <div style={styles.fieldGroup}>
                            <label style={styles.label}>4. Plage Temporelle (Optionnel)</label>
                            <div style={styles.dateGroup}>
                                <input
                                    type="date"
                                    value={startDate}
                                    onChange={(e) => setStartDate(e.target.value)}
                                    style={styles.input}
                                />
                                <input
                                    type="date"
                                    value={endDate}
                                    onChange={(e) => setEndDate(e.target.value)}
                                    style={styles.input}
                                />
                            </div>
                        </div>
                    </div>

                    <div style={{ marginTop: '20px' }}>
                        <button
                            type="submit"
                            disabled={loading || !selectedModelId || !selectedVersionId}
                            style={loading || !selectedModelId || !selectedVersionId ? styles.buttonDisabled : styles.buttonPrimary}
                        >
                            {loading ? 'Calcul des prédictions en cours...' : '⚡ Lancer la Prédiction & Afficher le Rapport'}
                        </button>
                    </div>
                </form>
            </div>

            {inferenceResult && dashboardMetrics && (
                <>
                    <div style={styles.mainNavTabs}>
                        <button
                            onClick={() => setActiveViewTab('dashboard')}
                            style={activeViewTab === 'dashboard' ? styles.mainNavActive : styles.mainNavInactive}
                        >
                            📈 1. Rapport & Graphiques
                        </button>
                        <button
                            onClick={() => setActiveViewTab('plotly')}
                            style={activeViewTab === 'plotly' ? styles.mainNavActive : styles.mainNavInactive}
                        >
                            📊 2. Graphique Interactif (Multi-Modèles)
                        </button>
                        <button
                            onClick={() => setActiveViewTab('action')}
                            style={activeViewTab === 'action' ? styles.mainNavActive : styles.mainNavInactive}
                        >
                            🛠️ 3. Actions & Comparaison BDD
                        </button>
                    </div>

                    {activeViewTab === 'dashboard' && (
                        <div>
                            <div style={styles.kpiGrid}>
                                <div style={styles.kpiCard}>
                                    <span style={styles.kpiTitle}>Nombre total d'échantillons</span>
                                    <span style={styles.kpiValue}>{dashboardMetrics.totalPoints}</span>
                                    <span style={styles.kpiSub}>Taux de complétion : <strong>{dashboardMetrics.completionRate}%</strong></span>
                                </div>

                                <div style={styles.kpiCard}>
                                    <span style={styles.kpiTitle}>Valeurs Manquantes (NULL)</span>
                                    <span style={{ ...styles.kpiValue, color: dashboardMetrics.missingCount > 0 ? '#d97706' : '#059669' }}>
                                        {dashboardMetrics.missingCount}
                                    </span>
                                    <span style={styles.kpiSub}>À imputer dans PostgreSQL</span>
                                </div>

                                <div style={styles.kpiCard}>
                                    <span style={styles.kpiTitle}>Moyenne (Prédite vs Réelle)</span>
                                    <span style={styles.kpiValue}>{dashboardMetrics.avgPredicted} <small style={{ fontSize: '12px' }}>{getUnitLabel(targetType)}</small></span>
                                    <span style={styles.kpiSub}>Réel : {dashboardMetrics.avgActual} {getUnitLabel(targetType)}</span>
                                </div>

                                <div style={styles.kpiCard}>
                                    <span style={styles.kpiTitle}>Score R²</span>
                                    <span style={{ ...styles.kpiValue, color: '#2563eb' }}>{dashboardMetrics.r2Score}%</span>
                                    <span style={styles.kpiSub}>Précision globale</span>
                                </div>
                            </div>

                            <div style={styles.card}>
                                <h3 style={styles.cardTitle}>📐 Métriques d'Erreur Théoriques</h3>
                                <div style={styles.metricRow}>
                                    <div style={styles.metricBox}>
                                        <span style={styles.metricLabel}>MAE (Erreur Absolue Moyenne)</span>
                                        <span style={styles.metricVal}>{dashboardMetrics.mae}</span>
                                        <span style={styles.metricUnit}>{getUnitLabel(targetType)}</span>
                                    </div>
                                    <div style={styles.metricBox}>
                                        <span style={styles.metricLabel}>RMSE (Racine de l'Erreur Quadratique)</span>
                                        <span style={styles.metricVal}>{dashboardMetrics.rmse}</span>
                                        <span style={styles.metricUnit}>{getUnitLabel(targetType)}</span>
                                    </div>
                                </div>
                            </div>

                            <div style={styles.card}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
                                    <h3 style={{ ...styles.cardTitle, margin: 0, border: 'none', padding: 0 }}>
                                        📊 Graphique Temporel Interactif (Prédictions vs Valeurs Réelles)
                                    </h3>

                                    <div style={{ display: 'flex', gap: '16px', background: '#f8fafc', padding: '6px 12px', borderRadius: '8px', border: '1px solid #cbd5e1' }}>
                                        <label style={{ fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                            <input
                                                type="checkbox"
                                                checked={showActual}
                                                onChange={(e) => setShowActual(e.target.checked)}
                                            />
                                            <span style={{ color: '#059669', fontWeight: 'bold' }}>🟢 Valeur Réelle</span>
                                        </label>
                                        <label style={{ fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                            <input
                                                type="checkbox"
                                                checked={showPredicted}
                                                onChange={(e) => setShowPredicted(e.target.checked)}
                                            />
                                            <span style={{ color: '#2563eb', fontWeight: 'bold' }}>🔵 Prédiction</span>
                                        </label>
                                    </div>
                                </div>

                                <p style={{ fontSize: '12px', color: '#64748b', marginTop: 0 }}>
                                    💡 <strong>Astuce d'interaction :</strong> Utilisez la barre coulissante sous le graphique <em>(Brush Slider)</em> pour zoomer ou vous déplacer dans le temps.
                                </p>

                                <div style={{ width: '100%', height: 460, marginTop: '16px' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={predictionList} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />

                                            <XAxis
                                                dataKey="datetime"
                                                stroke="#64748b"
                                                fontSize={11}
                                                tickLine={false}
                                                tickFormatter={formatXAxisDate}
                                                minTickGap={30}
                                                dy={6}
                                            />

                                            <YAxis stroke="#64748b" fontSize={12} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #cbd5e1', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                                formatter={(value, name) => [`${value} ${getUnitLabel(targetType)}`, name]}
                                            />

                                            <Legend verticalAlign="top" align="right" height={36} wrapperStyle={{ paddingTop: '0px' }} />

                                            {showActual && (
                                                <Line
                                                    type="monotone"
                                                    dataKey="actual"
                                                    name={`Valeur Réelle (${getUnitLabel(targetType)})`}
                                                    stroke="#059669"
                                                    strokeWidth={2.5}
                                                    dot={{ r: 1.5 }}
                                                    activeDot={{ r: 6 }}
                                                    connectNulls
                                                />
                                            )}

                                            {showPredicted && (
                                                <Line
                                                    type="monotone"
                                                    dataKey="predicted"
                                                    name={`Prédiction (${getUnitLabel(targetType)})`}
                                                    stroke="#2563eb"
                                                    strokeWidth={2}
                                                    strokeDasharray="4 4"
                                                    dot={{ r: 1.5 }}
                                                    activeDot={{ r: 6 }}
                                                />
                                            )}

                                            <Brush
                                                dataKey="datetime"
                                                height={30}
                                                stroke="#2563eb"
                                                fillColor="#f8fafc"
                                                tickFormatter={formatXAxisDate}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>
                    )}

                    {activeViewTab === 'plotly' && (
                        <div style={styles.card}>
                            <h3 style={styles.cardTitle}>✨ Graphique Interactif Plotly (Comparaison Multi-Modèles)</h3>

                            <div style={{ marginBottom: '16px', padding: '12px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #cbd5e1' }}>
                                <label style={{ ...styles.label, marginBottom: '8px', display: 'block' }}>
                                    Modèles à comparer pour {selectedAttraction} (filtrés par thuộc tính mục tiêu tương đồng) :
                                </label>
                                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                                    {comparableModels.map((m) => (
                                        <label key={m.model_id} style={styles.radioLabel}>
                                            <input
                                                type="checkbox"
                                                checked={selectedCompareModels.includes(m.model_id)}
                                                onChange={() => handleToggleCompareModel(m.model_id)}
                                            />
                                            <strong>{m.model_id}</strong>
                                            {m.target_column && (
                                                <span style={{ fontSize: '11px', color: '#64748b', marginLeft: '4px' }}>
                                                    ({m.target_column})
                                                </span>
                                            )}
                                        </label>
                                    ))}
                                </div>
                            </div>

                            {loadingDashboardHtml ? (
                                <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                                    ⏳ Chargement du graphique Plotly...
                                </div>
                            ) : htmlDashboard ? (
                                <iframe
                                    title="Graphique interactif Plotly"
                                    srcDoc={htmlDashboard}
                                    style={{
                                        width: '100%',
                                        height: '750px',
                                        border: '1px solid #e2e8f0',
                                        borderRadius: '8px'
                                    }}
                                />
                            ) : (
                                <div style={{ textAlign: 'center', padding: '40px', color: '#94a3b8' }}>
                                    Aucune donnée disponible pour le graphique.
                                </div>
                            )}
                        </div>
                    )}

                    {activeViewTab === 'action' && (
                        <div style={styles.card}>
                            <div style={styles.actionTabHeader}>
                                <button
                                    onClick={() => setActionTab('impute')}
                                    style={actionTab === 'impute' ? styles.actionTabActive : styles.actionTabInactive}
                                >
                                    📊 1. Tableau Détaillé & Écarts ({selectedAttraction})
                                </button>
                                <button
                                    onClick={() => setActionTab('export_db')}
                                    style={actionTab === 'export_db' ? styles.actionTabActive : styles.actionTabInactive}
                                >
                                    📦 2. Exporter la Table vers PostgreSQL
                                </button>
                                <button
                                    onClick={() => setActionTab('export_file')}
                                    style={actionTab === 'export_file' ? styles.actionTabActive : styles.actionTabInactive}
                                >
                                    📁 3. Exporter CSV / Excel
                                </button>
                            </div>

                            {actionTab === 'impute' && (
                                <div>
                                    <div style={styles.hyperContainer}>
                                        <div style={{ display: 'flex', gap: '24px', alignItems: 'center', flexWrap: 'wrap' }}>
                                            <div>
                                                <label style={styles.subLabel}>Table Impute ({selectedAttraction}):</label>
                                                <select
                                                    value={selectedVersionId}
                                                    onChange={(e) => setSelectedVersionId(e.target.value)}
                                                    style={{ ...styles.select, width: '260px', marginTop: '4px' }}
                                                >
                                                    {allAttractionVersions.length > 0 ? (
                                                        allAttractionVersions.map((verName) => (
                                                            <option key={verName} value={verName}>
                                                                {verName}
                                                            </option>
                                                        ))
                                                    ) : (
                                                        <option value="">Aucune table disponible</option>
                                                    )}
                                                </select>
                                            </div>

                                            <div>
                                                <label style={styles.subLabel}>Colonne Cible :</label>
                                                <input
                                                    type="text"
                                                    value={targetColumn}
                                                    onChange={(e) => setTargetColumn(e.target.value)}
                                                    style={{ ...styles.input, width: '200px', marginTop: '4px' }}
                                                />
                                            </div>

                                            <div>
                                                <label style={styles.subLabel}>Mode d'Imputation BDD :</label>
                                                <div style={{ display: 'flex', gap: '16px', marginTop: '6px' }}>
                                                    <label style={styles.radioLabel}>
                                                        <input
                                                            type="radio"
                                                            name="impMode"
                                                            value="fill_missing"
                                                            checked={imputationMode === 'fill_missing'}
                                                            onChange={() => setImputationMode('fill_missing')}
                                                        /> 🧩 Remplir uniquement les valeurs manquantes (fill_missing)
                                                    </label>
                                                    <label style={styles.radioLabel}>
                                                        <input
                                                            type="radio"
                                                            name="impMode"
                                                            value="overwrite"
                                                            checked={imputationMode === 'overwrite'}
                                                            onChange={() => setImputationMode('overwrite')}
                                                        /> 🔄 Écraser la colonne entière (overwrite)
                                                    </label>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                                        <label style={styles.radioLabel}>
                                            <input
                                                type="checkbox"
                                                checked={filterMissingOnly}
                                                onChange={(e) => setFilterMissingOnly(e.target.checked)}
                                            /> ⚠️ Afficher uniquement les valeurs manquantes (NULL)
                                        </label>

                                        <button
                                            onClick={handleApplyImputation}
                                            disabled={applyingDb}
                                            style={applyingDb ? styles.buttonDisabled : styles.buttonSuccess}
                                        >
                                            {applyingDb ? 'Mise à jour BDD...' : '💾 Appliquer l’Imputation dans la BDD'}
                                        </button>
                                    </div>

                                    <div style={{ overflowX: 'auto', maxHeight: '420px' }}>
                                        <table style={styles.table}>
                                            <thead>
                                                <tr style={styles.thRow}>
                                                    <th style={styles.th}>Horodatage (Datetime)</th>
                                                    <th style={styles.th}>Mois-Année (MM/YYYY)</th>
                                                    <th style={styles.th}>Valeur Réelle</th>
                                                    <th style={styles.th}>Valeur Prédite</th>
                                                    <th style={styles.th}>Écart Absolu</th>
                                                    <th style={styles.th}>Écart (%)</th>
                                                    <th style={styles.th}>Statut</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {displayedPredictions.length === 0 ? (
                                                    <tr>
                                                        <td colSpan="7" style={styles.tdEmpty}>Aucun enregistrement à afficher pour {selectedAttraction}.</td>
                                                    </tr>
                                                ) : (
                                                    displayedPredictions.map((row, idx) => {
                                                        const isMissing = row.actual === null || row.actual === undefined || isNaN(row.actual);

                                                        let monthYear = 'N/A';
                                                        if (row.datetime) {
                                                            const d = new Date(row.datetime);
                                                            if (!isNaN(d.getTime())) {
                                                                const month = String(d.getMonth() + 1).padStart(2, '0');
                                                                const year = d.getFullYear();
                                                                monthYear = `${month}/${year}`;
                                                            }
                                                        }

                                                        const actualVal = Number(row.actual);
                                                        const predVal = Number(row.predicted);
                                                        const diff = !isMissing ? Math.abs(actualVal - predVal) : null;

                                                        let percentError = 'N/A';
                                                        if (!isMissing && actualVal !== 0) {
                                                            const p = (diff / actualVal) * 100;
                                                            percentError = `${p.toFixed(2)} %`;
                                                        } else if (!isMissing && actualVal === 0) {
                                                            percentError = '0.00 %';
                                                        }

                                                        return (
                                                            <tr key={(row.datetime || row.horodatage || idx) + (row.id_attraction || '')} style={styles.tr}>
                                                                <td style={{ ...styles.td, fontFamily: 'monospace' }}>
                                                                    {formatDateTimeStandard(row.datetime || row.horodatage)}
                                                                </td>
                                                                <td style={{ ...styles.td, fontWeight: '600', color: '#475569' }}>{monthYear}</td>
                                                                <td style={styles.td}>
                                                                    {isMissing ? (
                                                                        <span style={styles.badgeNull}>Manquant (NULL)</span>
                                                                    ) : (
                                                                        actualVal
                                                                    )}
                                                                </td>
                                                                <td style={styles.td}>
                                                                    <input
                                                                        type="number"
                                                                        step="0.01"
                                                                        value={row.predicted}
                                                                        onChange={(e) => handleValueChange(idx, e.target.value)}
                                                                        style={styles.tableInput}
                                                                    />
                                                                </td>
                                                                <td style={styles.td}>
                                                                    {!isMissing ? diff.toFixed(2) : '-'}
                                                                </td>
                                                                <td style={{ ...styles.td, fontWeight: '700', color: !isMissing && parseFloat(percentError) > 15 ? '#b91c1c' : '#059669' }}>
                                                                    {percentError}
                                                                </td>
                                                                <td style={styles.td}>
                                                                    {isMissing ? (
                                                                        <span style={{ color: '#d97706', fontWeight: '600' }}>⚠️ À imputer</span>
                                                                    ) : (
                                                                        <span style={{ color: '#059669' }}>OK</span>
                                                                    )}
                                                                </td>
                                                            </tr>
                                                        );
                                                    })
                                                )}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            )}

                            {actionTab === 'export_db' && (
                                <div style={styles.exportContainer}>
                                    <p style={styles.exportInfoText}>
                                        💡 <strong>Information :</strong> Génération d'une nouvelle table exportée pour l'attraction{' '}
                                        <mark style={{ backgroundColor: '#fef08a', padding: '2px 6px', borderRadius: '4px' }}>{selectedAttraction}</mark>.
                                        La nouvelle table inclura les colonnes <code style={styles.codeInline}>predicted_value</code>,{' '}
                                        <code style={styles.codeInline}>residual_error</code> et <code style={styles.codeInline}>prediction_created_at</code>.
                                    </p>

                                    <div style={styles.configGrid}>
                                        <div style={styles.fieldGroup}>
                                            <label style={styles.label}>1. Schéma Cible</label>
                                            <input
                                                type="text"
                                                value={targetSchema}
                                                onChange={(e) => setTargetSchema(e.target.value)}
                                                placeholder="ex: predictions"
                                                style={styles.input}
                                            />
                                        </div>

                                        <div style={styles.fieldGroup}>
                                            <label style={styles.label}>2. Nom de la Table Exportée</label>
                                            <input
                                                type="text"
                                                value={outputTableName}
                                                onChange={(e) => setOutputTableName(e.target.value)}
                                                placeholder="Nom de la table"
                                                style={styles.input}
                                            />
                                        </div>

                                        <div style={styles.fieldGroup}>
                                            <label style={styles.label}>3. Gestion des Doublons</label>
                                            <select
                                                value={ifExistsMode}
                                                onChange={(e) => setIfExistsMode(e.target.value)}
                                                style={styles.select}
                                            >
                                                <option value="replace">🔄 Remplacer la table (Replace)</option>
                                                <option value="append">➕ Ajouter aux données (Append)</option>
                                                <option value="fail">❌ Annuler si existante (Fail)</option>
                                            </select>
                                        </div>
                                    </div>

                                    <button
                                        onClick={handleExportToSchema}
                                        disabled={exportingSchema}
                                        style={exportingSchema ? styles.buttonDisabled : styles.buttonPurple}
                                    >
                                        {exportingSchema ? 'Création de la table...' : `🚀 Exporter vers le schéma '${targetSchema}'`}
                                    </button>
                                </div>
                            )}

                            {actionTab === 'export_file' && (
                                <div style={styles.exportContainer}>
                                    <p style={styles.exportInfoText}>
                                        📥 <strong>Exporter les prédictions via le Service Router :</strong> Génère un fichier contenant toutes les colonnes sources de la table <code style={styles.codeInline}>{selectedVersionId}</code> ainsi que les colonnes de prédiction correspondantes aux modèles sélectionnés.
                                    </p>

                                    <div style={{ display: 'flex', gap: '16px', marginTop: '20px', flexWrap: 'wrap' }}>
                                        <button
                                            onClick={() => handleExportFile('csv')}
                                            disabled={exportingFile}
                                            style={{ ...(exportingFile ? styles.buttonDisabled : styles.buttonPrimary), backgroundColor: exportingFile ? '#94a3b8' : '#2563eb', flex: 1, minWidth: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                                        >
                                            📄 {exportingFile ? 'Exportation en cours...' : 'Exporter en CSV'}
                                        </button>

                                        <button
                                            onClick={() => handleExportFile('xlsx')}
                                            disabled={exportingFile}
                                            style={{ ...(exportingFile ? styles.buttonDisabled : styles.buttonPrimary), backgroundColor: exportingFile ? '#94a3b8' : '#16a34a', flex: 1, minWidth: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                                        >
                                            📊 {exportingFile ? 'Exportation en cours...' : 'Exporter en Excel (.xlsx)'}
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

// ==================== STYLES TỐI ƯU CSS-IN-JS ====================
const styles = {
    container: {
        padding: '24px',
        maxWidth: '1360px',
        margin: '0 auto',
        fontFamily: "'Inter', sans-serif",
        color: '#1e293b',
        boxSizing: 'border-box'
    },
    header: { marginBottom: '20px', borderBottom: '1px solid #e2e8f0', paddingBottom: '16px' },
    title: { margin: '0 0 8px 0', color: '#0f172a', fontSize: '22px', fontWeight: '700' },
    subtitle: { margin: 0, color: '#64748b', fontSize: '14px' },

    tabContainer: { display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap' },
    tabActive: { padding: '10px 18px', backgroundColor: '#2563eb', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '13px' },
    tabInactive: { padding: '10px 18px', backgroundColor: '#f1f5f9', color: '#475569', border: '1px solid #cbd5e1', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '13px' },

    card: {
        background: '#ffffff',
        borderRadius: '12px',
        padding: '24px',
        marginBottom: '24px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
        border: '1px solid #e2e8f0',
        boxSizing: 'border-box',
        overflow: 'hidden'
    },
    cardTitle: { margin: '0 0 16px 0', fontSize: '16px', fontWeight: '600', color: '#334155', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' },

    configGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: '20px',
        width: '100%',
        boxSizing: 'border-box'
    },
    fieldGroup: { display: 'flex', flexDirection: 'column', gap: '8px', minWidth: 0 },
    dateGroup: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', width: '100%', boxSizing: 'border-box' },

    label: { fontSize: '13px', fontWeight: '600', color: '#475569', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
    subLabel: { fontSize: '12px', fontWeight: '600', color: '#64748b' },
    radioLabel: { fontSize: '13px', cursor: 'pointer', color: '#334155', display: 'flex', alignItems: 'center', gap: '6px' },

    select: {
        height: '42px',
        padding: '0 12px',
        borderRadius: '8px',
        border: '1px solid #cbd5e1',
        fontSize: '14px',
        backgroundColor: '#f8fafc',
        width: '100%',
        boxSizing: 'border-box'
    },
    input: {
        height: '42px',
        padding: '0 12px',
        borderRadius: '8px',
        border: '1px solid #cbd5e1',
        fontSize: '14px',
        backgroundColor: '#ffffff',
        width: '100%',
        boxSizing: 'border-box'
    },

    mainNavTabs: { display: 'flex', gap: '8px', marginBottom: '20px', borderBottom: '2px solid #cbd5e1', paddingBottom: '2px', flexWrap: 'wrap' },
    mainNavActive: { padding: '12px 24px', backgroundColor: '#0f172a', color: '#ffffff', border: 'none', borderRadius: '8px 8px 0 0', fontWeight: '700', cursor: 'pointer', fontSize: '14px' },
    mainNavInactive: { padding: '12px 24px', backgroundColor: '#e2e8f0', color: '#475569', border: 'none', borderRadius: '8px 8px 0 0', fontWeight: '600', cursor: 'pointer', fontSize: '14px' },

    kpiGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '24px' },
    kpiCard: { background: '#ffffff', border: '1px solid #e2e8f0', padding: '18px', borderRadius: '10px', display: 'flex', flexDirection: 'column', gap: '6px', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' },
    kpiTitle: { fontSize: '12px', color: '#64748b', fontWeight: '600', textTransform: 'uppercase' },
    kpiValue: { fontSize: '24px', fontWeight: '800', color: '#0f172a' },
    kpiSub: { fontSize: '12px', color: '#475569' },

    metricRow: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' },
    metricBox: { padding: '16px', backgroundColor: '#f8fafc', borderRadius: '8px', border: '1px solid #cbd5e1', textAlign: 'center' },
    metricLabel: { display: 'block', fontSize: '12px', fontWeight: '600', color: '#64748b', marginBottom: '6px' },
    metricVal: { display: 'block', fontSize: '22px', fontWeight: '800', color: '#1e293b' },
    metricUnit: { fontSize: '11px', color: '#94a3b8' },

    actionTabHeader: { display: 'flex', gap: '12px', borderBottom: '2px solid #e2e8f0', marginBottom: '20px', paddingBottom: '2px', flexWrap: 'wrap' },
    actionTabActive: { padding: '12px 20px', backgroundColor: '#ffffff', color: '#2563eb', border: 'none', borderBottom: '3px solid #2563eb', fontWeight: '700', cursor: 'pointer', fontSize: '14px' },
    actionTabInactive: { padding: '12px 20px', backgroundColor: 'transparent', color: '#64748b', border: 'none', fontWeight: '600', cursor: 'pointer', fontSize: '14px' },

    hyperContainer: { padding: '16px', backgroundColor: '#f1f5f9', borderRadius: '10px', marginBottom: '16px', border: '1px solid #cbd5e1' },
    exportContainer: { padding: '20px', backgroundColor: '#f8fafc', borderRadius: '10px', border: '1px solid #e2e8f0' },
    exportInfoText: { fontSize: '13px', color: '#334155', lineHeight: '1.6', marginBottom: '20px', marginTop: 0 },
    codeInline: { backgroundColor: '#e2e8f0', color: '#0f172a', padding: '2px 6px', borderRadius: '4px', fontFamily: 'monospace', fontSize: '12px', margin: '0 3px' },

    buttonPrimary: { width: '100%', padding: '12px 20px', backgroundColor: '#2563eb', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '14px', boxSizing: 'border-box' },
    buttonSuccess: { padding: '10px 18px', backgroundColor: '#059669', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '13px' },
    buttonPurple: { width: '100%', padding: '12px 20px', backgroundColor: '#7c3aed', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '14px', boxSizing: 'border-box' },
    buttonDisabled: { width: '100%', padding: '12px 20px', backgroundColor: '#94a3b8', color: '#ffffff', border: 'none', borderRadius: '8px', cursor: 'not-allowed', fontSize: '14px', boxSizing: 'border-box' },

    errorAlert: { backgroundColor: '#fef2f2', color: '#b91c1c', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #fecaca', fontSize: '14px' },
    successAlert: { backgroundColor: '#ecfdf5', color: '#047857', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #a7f3d0', fontSize: '14px' },

    table: { width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' },
    thRow: { backgroundColor: '#f8fafc', borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, zIndex: 10 },
    th: { padding: '10px 12px', fontWeight: '600', color: '#475569' },
    tr: { borderBottom: '1px solid #f1f5f9' },
    td: { padding: '8px 12px', color: '#334155' },
    tdEmpty: { padding: '16px', textAlign: 'center', color: '#94a3b8' },

    tableInput: { height: '32px', padding: '0 8px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '13px', width: '110px', fontWeight: '600', color: '#2563eb' },
    badgeNull: { padding: '2px 8px', backgroundColor: '#fef3c7', color: '#d97706', borderRadius: '6px', fontSize: '11px', fontWeight: '700' }
};