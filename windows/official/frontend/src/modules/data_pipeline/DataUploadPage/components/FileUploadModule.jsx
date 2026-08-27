import React, { useState, useRef } from 'react';
import axios from 'axios';
import {
    UploadCloud,
    CheckCircle,
    AlertCircle,
    Loader2,
    FileText,
    X,
    RotateCcw
} from 'lucide-react';

// URL Backend FastAPI
const API_BASE_URL = 'http://127.0.0.1:8000';

export default function FileUploadModule({ onUploadSuccess }) {
    const [files, setFiles] = useState({
        // 6 file bắt buộc (Bán động/Thời gian thực)
        visitor_file: null,
        etat_file: null,
        weather_file: null,
        horaire_file: null,
        elec_file: null,
        ec_file: null,
        // 2 file tĩnh (Tùy chọn)
        cadence_file: null,
        surface_file: null,
    });

    const [loading, setLoading] = useState(false);
    const [response, setResponse] = useState(null);
    const [error, setError] = useState(null);

    // Ref lưu trữ các input DOM để có thể reset value thủ công khi xóa
    const inputRefs = useRef({});

    // Cấu hình danh sách file, phân loại isRequired
    const fileInputs = [
        { key: 'visitor_file', label: '1. Fichier Visiteurs', desc: 'Données de fréquentation', isRequired: true },
        { key: 'etat_file', label: '2. Fichier États', desc: 'États opérationnels', isRequired: true },
        { key: 'weather_file', label: '3. Fichier Météo', desc: 'Température, humidité', isRequired: true },
        { key: 'horaire_file', label: '4. Fichier Horaires', desc: 'Calendrier d’ouverture', isRequired: true },
        { key: 'elec_file', label: '5. Fichier Électricité', desc: 'Consommation électrique', isRequired: true },
        { key: 'ec_file', label: '6. Fichier Énergie Thermique', desc: 'Consommation chaud/froid', isRequired: true },
        { key: 'cadence_file', label: '7. Fichier Cadence (Optionnel)', desc: 'Capacité & cadence des jeux', isRequired: false },
        { key: 'surface_file', label: '8. Fichier Surface (Optionnel)', desc: 'Superficie des bâtiments', isRequired: false },
    ];

    const formatFileSize = (bytes) => {
        if (!bytes) return '';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    const handleFileChange = (key, selectedFile) => {
        if (selectedFile) {
            setFiles((prev) => ({ ...prev, [key]: selectedFile }));
            setError(null);
        }
    };

    const handleRemoveFile = (key, e) => {
        if (e) {
            e.stopPropagation();
            e.preventDefault();
        }
        setFiles((prev) => ({ ...prev, [key]: null }));
        // Reset HTML input element
        if (inputRefs.current[key]) {
            inputRefs.current[key].value = '';
        }
    };

    const handleResetAll = () => {
        setFiles({
            visitor_file: null,
            etat_file: null,
            weather_file: null,
            horaire_file: null,
            elec_file: null,
            ec_file: null,
            cadence_file: null,
            surface_file: null,
        });
        // Reset toàn bộ input file refs
        Object.keys(inputRefs.current).forEach((key) => {
            if (inputRefs.current[key]) inputRefs.current[key].value = '';
        });
        setResponse(null);
        setError(null);
    };

    // Chỉ kiểm tra các file BẮT BUỘC (isRequired: true)
    const isRequiredFilesSelected = fileInputs
        .filter((item) => item.isRequired)
        .every((item) => files[item.key] !== null);

    const requiredCount = fileInputs.filter((item) => item.isRequired && files[item.key] !== null).length;
    const optionalCount = fileInputs.filter((item) => !item.isRequired && files[item.key] !== null).length;

    // HÀM XỬ LÝ UPLOAD ĐÃ ĐƯỢC ĐỔI THỨ TỰ API CHUẨN
    const handleUpload = async () => {
        if (!isRequiredFilesSelected) return;

        setLoading(true);
        setError(null);
        setResponse(null);

        try {
            let staticMsg = "";

            // 🚀 1. UPLOAD FILE TĨNH TRƯỚC (NẾU CÓ)
            // Phải chạy trước để nạp/xóa/cập nhật dim_attraction & dim_cadence trước khi nạp Fact!
            if (files.cadence_file || files.surface_file) {
                const staticFormData = new FormData();
                if (files.cadence_file) staticFormData.append('cadence_file', files.cadence_file);
                if (files.surface_file) staticFormData.append('surface_file', files.surface_file);

                await axios.post(`${API_BASE_URL}/etl/upload-static-files`, staticFormData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                });
                staticMsg = " (incluant métadonnées statiques)";
            }

            // 🚀 2. UPLOAD 6 FILE ĐỘNG SAU (Nạp dữ liệu Fact khi Dimension đã hoàn chỉnh)
            const dynamicFormData = new FormData();
            dynamicFormData.append('visitor_file', files.visitor_file);
            dynamicFormData.append('etat_file', files.etat_file);
            dynamicFormData.append('weather_file', files.weather_file);
            dynamicFormData.append('horaire_file', files.horaire_file);
            dynamicFormData.append('elec_file', files.elec_file);
            dynamicFormData.append('ec_file', files.ec_file);

            const resDynamic = await axios.post(`${API_BASE_URL}/etl/upload-6-files`, dynamicFormData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            // Thông báo thành công
            setResponse({
                message: resDynamic.data?.message || `Téléchargement et traitement des données${staticMsg} réussis !`
            });

            if (onUploadSuccess) {
                onUploadSuccess('fact_attraction_hourly');
            }
        } catch (err) {
            console.error("Lỗi Upload chi tiết:", err);
            const detailMsg = err.response?.data?.detail;
            setError(typeof detailMsg === 'string' ? detailMsg : 'Erreur lors de la connexion au serveur backend !');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ marginBottom: '40px', paddingBottom: '24px', borderBottom: '2px solid #e2e8f0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <h2 style={{ color: '#1e293b', margin: 0, fontSize: '20px', fontWeight: '700' }}>
                    Module 1 : Pipeline de Données & Ingestion
                </h2>
                {(requiredCount > 0 || optionalCount > 0) && (
                    <button
                        onClick={handleResetAll}
                        disabled={loading}
                        style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '6px',
                            background: 'transparent',
                            border: '1px solid #cbd5e1',
                            borderRadius: '6px',
                            padding: '6px 12px',
                            fontSize: '13px',
                            color: '#64748b',
                            cursor: loading ? 'not-allowed' : 'pointer',
                        }}
                    >
                        <RotateCcw size={14} /> Réinitialiser
                    </button>
                )}
            </div>

            <p style={{ color: '#64748b', marginBottom: '24px', fontSize: '14px', lineHeight: '1.5' }}>
                Chargez les 6 fichiers de données temporelles obligatoires. Les fichiers de métadonnées (Cadence, Surface) sont optionnels.
                <span style={{ marginLeft: '8px', fontWeight: '600', color: isRequiredFilesSelected ? '#16a34a' : '#d97706' }}>
                    ({requiredCount}/6 obligatoires {optionalCount > 0 ? `+ ${optionalCount} optionnels` : ''})
                </span>
            </p>

            {/* Grid ô chọn file */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '24px' }}>
                {fileInputs.map((item) => {
                    const currentFile = files[item.key];
                    return (
                        <div
                            key={item.key}
                            style={{
                                border: currentFile
                                    ? '2px solid #22c55e'
                                    : item.isRequired
                                        ? '2px dashed #cbd5e1'
                                        : '2px dashed #e2e8f0',
                                borderRadius: '10px',
                                padding: '16px',
                                backgroundColor: currentFile ? '#f0fdf4' : item.isRequired ? '#f8fafc' : '#f1f5f9',
                                opacity: !item.isRequired && !currentFile ? 0.85 : 1,
                                display: 'flex',
                                flexDirection: 'column',
                                justifyContent: 'space-between',
                            }}
                        >
                            <div>
                                <div style={{ fontWeight: '600', fontSize: '14px', color: '#334155', marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                                    <span>{item.label}</span>
                                    {!item.isRequired && <span style={{ fontSize: '11px', color: '#64748b', fontStyle: 'italic' }}>Optionnel</span>}
                                </div>
                                <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '14px' }}>
                                    {item.desc}
                                </div>
                            </div>

                            {currentFile ? (
                                <div
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'space-between',
                                        padding: '8px 12px',
                                        backgroundColor: '#ffffff',
                                        border: '1px solid #bbf7d0',
                                        borderRadius: '6px',
                                    }}
                                >
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                                        <FileText size={18} color="#16a34a" style={{ flexShrink: 0 }} />
                                        <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                            <div style={{ fontSize: '13px', fontWeight: '600', color: '#15803d' }}>
                                                {currentFile.name}
                                            </div>
                                            <div style={{ fontSize: '11px', color: '#16a34a' }}>
                                                {formatFileSize(currentFile.size)}
                                            </div>
                                        </div>
                                    </div>
                                    <button
                                        type="button"
                                        onClick={(e) => handleRemoveFile(item.key, e)}
                                        style={{
                                            background: 'none',
                                            border: 'none',
                                            cursor: 'pointer',
                                            color: '#ef4444',
                                            padding: '4px',
                                            display: 'flex',
                                            alignItems: 'center',
                                        }}
                                    >
                                        <X size={16} />
                                    </button>
                                </div>
                            ) : (
                                <label
                                    style={{
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '8px',
                                        padding: '8px 14px',
                                        backgroundColor: item.isRequired ? '#0284c7' : '#64748b',
                                        color: '#ffffff',
                                        borderRadius: '6px',
                                        fontSize: '13px',
                                        fontWeight: '500',
                                        cursor: 'pointer',
                                    }}
                                >
                                    <FileText size={16} />
                                    Sélectionner
                                    <input
                                        ref={(el) => (inputRefs.current[item.key] = el)}
                                        type="file"
                                        accept=".csv, .xlsx, .xls"
                                        style={{ display: 'none' }}
                                        onChange={(e) => handleFileChange(item.key, e.target.files[0])}
                                    />
                                </label>
                            )}
                        </div>
                    );
                })}
            </div>

            <button
                onClick={handleUpload}
                disabled={!isRequiredFilesSelected || loading}
                style={{
                    width: '100%',
                    padding: '14px',
                    backgroundColor: !isRequiredFilesSelected || loading ? '#94a3b8' : '#2563eb',
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '16px',
                    fontWeight: '600',
                    cursor: !isRequiredFilesSelected || loading ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '10px',
                }}
            >
                {loading ? (
                    <>
                        <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
                        Traitement des données...
                    </>
                ) : (
                    <>
                        <UploadCloud size={20} /> Importer & Traiter les Données
                    </>
                )}
            </button>

            {response && (
                <div style={{ marginTop: '20px', padding: '16px', backgroundColor: '#dcfce7', borderRadius: '8px', color: '#15803d', border: '1px solid #86efac' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <CheckCircle size={22} style={{ flexShrink: 0 }} />
                        <strong style={{ fontSize: '15px' }}>{response.message}</strong>
                    </div>
                </div>
            )}

            {error && (
                <div style={{ marginTop: '20px', padding: '16px', backgroundColor: '#fee2e2', borderRadius: '8px', color: '#b91c1c', border: '1px solid #fca5a5', display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <AlertCircle size={22} style={{ flexShrink: 0 }} />
                    <div style={{ fontSize: '14px' }}>
                        <strong>Échec :</strong> {error}
                    </div>
                </div>
            )}

            <style>{`
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    );
}