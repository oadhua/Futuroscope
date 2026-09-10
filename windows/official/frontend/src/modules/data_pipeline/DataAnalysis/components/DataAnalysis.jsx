import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
    BarChart3, Database, AlertTriangle, Layers,
    Filter, ArrowRight, Search, Table, Activity, Eye, EyeOff, ChevronDown,
    TrendingDown, Copy, GitCompare, PieChart as PieIcon, ChevronLeft, ChevronRight
} from 'lucide-react';
import {
    PieChart, Pie, Cell, Tooltip,
    BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Legend
} from 'recharts';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#06B6D4', '#64748b'];

// Hàm tự động tạo màu phân bố đều theo không gian HSL cho vô số version (đồng bộ với SingleColumn)
const getDynamicColor = (index, total) => {
    if (total <= 1) return '#059669';
    const hue = Math.round((index * 360) / total);
    return `hsl(${hue}, 70%, 45%)`;
};

export default function DataAnalysis() {
    // Mode d'affichage: 'single' (Analyse simple) ou 'compare' (Comparaison)
    const [viewMode, setViewMode] = useState('single');

    // États du mode simple
    const [selectedVersion, setSelectedVersion] = useState('v0_raw');
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [data, setData] = useState(null);

    // États du mode comparaison
    const [v1Version, setV1Version] = useState('v0_raw');
    const [v2Version, setV2Version] = useState('');
    const [compareData, setCompareData] = useState(null);

    // Listes issues du schéma data_prep
    const [allVersions, setAllVersions] = useState(['v0_raw']);
    const [attractionList, setAttractionList] = useState([]);

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Onglets et filtres
    const [activeTab, setActiveTab] = useState('overview');

    // Affichage complet ou réduit des graphiques
    const [showAllPie, setShowAllPie] = useState(true);
    const [showAllBar, setShowAllBar] = useState(true);

    const [searchTerm, setSearchTerm] = useState('');
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

    // Pagination
    const [currentPage, setCurrentPage] = useState(1);
    const [rowsPerPage, setRowsPerPage] = useState(10);

    // Ref pour annuler les requêtes précédentes en cas de clics rapides
    const abortControllerRef = useRef(null);

    // 1. Filtrer la liste des versions selon l'attraction sélectionnée
    const filteredVersionList = useMemo(() => {
        if (!selectedAttraction || selectedAttraction === 'ALL') {
            return allVersions;
        }
        const attrUpper = selectedAttraction.toUpperCase();
        return allVersions.filter(v => {
            if (v === 'v0_raw') return true;
            return v.toUpperCase().includes(attrUpper);
        });
    }, [allVersions, selectedAttraction]);

    // Gestion du changement d'attraction
    const handleAttractionChange = (newAttr) => {
        setSelectedAttraction(newAttr);

        let newList = allVersions;
        if (newAttr && newAttr !== 'ALL') {
            const attrUpper = newAttr.toUpperCase();
            newList = allVersions.filter(v => v === 'v0_raw' || v.toUpperCase().includes(attrUpper));
        }

        if (!newList.includes(selectedVersion)) {
            setSelectedVersion(newList[0] || 'v0_raw');
        }

        let currentV1 = v1Version;
        if (!newList.includes(v1Version)) {
            currentV1 = 'v0_raw';
            setV1Version('v0_raw');
        }

        if (!newList.includes(v2Version)) {
            const v1Idx = newList.indexOf(currentV1);
            if (v1Idx !== -1 && v1Idx + 1 < newList.length) {
                setV2Version(newList[v1Idx + 1]);
            } else {
                setV2Version(newList[newList.length - 1] || 'v0_raw');
            }
        }
    };

    // 2. Charger toutes les versions enregistrées dans le schéma data_prep
    const fetchVersions = async () => {
        try {
            const res = await fetch('http://localhost:8000/analysis/versions-schema');
            if (res.ok) {
                const resData = await res.json();
                const list = resData.versions || resData || [];
                const vIds = list.map(v => typeof v === 'string' ? v : (v.version_id || v.table_name));

                if (!vIds.includes('v0_raw')) {
                    vIds.unshift('v0_raw');
                }

                const uniqueVersions = [...new Set(vIds)];
                setAllVersions(uniqueVersions);

                if (uniqueVersions.length > 1 && !v2Version) {
                    setV2Version(uniqueVersions[1]);
                }
            }
        } catch (e) {
            console.error("Erreur lors de la récupération des versions du schéma data_prep:", e);
        }
    };

    // 3. Charger le profilage d'une seule version
    const fetchData = async (verId, attrId, signal) => {
        setLoading(true);
        setError(null);
        try {
            let url = `http://localhost:8000/analysis/analyse-globale?version_id=${verId}&nb_lignes_apercu=100`;
            if (attrId && attrId !== 'ALL') {
                url += `&id_attraction=${attrId}`;
            }

            const response = await fetch(url, { signal });
            if (!response.ok) throw new Error(`Erreur HTTP: ${response.status}`);
            const resData = await response.json();

            setData(resData);

            if (resData.liste_attractions && resData.liste_attractions.length > 0) {
                setAttractionList(prev => prev.length === 0 ? resData.liste_attractions : prev);
            }
            setCurrentPage(1);
        } catch (err) {
            if (err.name !== 'AbortError') {
                setError(err.message || 'Impossible de charger les données du profilage');
            }
        } finally {
            if (!signal.aborted) setLoading(false);
        }
    };

    // 4. Charger la comparaison entre deux versions
    const fetchComparison = async (v1, v2, attrId, signal) => {
        setLoading(true);
        setError(null);
        try {
            let url = `http://localhost:8000/analysis/compare-versions?v1_version_id=${v1}&v2_version_id=${v2}`;
            if (attrId && attrId !== 'ALL') {
                url += `&id_attraction=${attrId}`;
            }

            const response = await fetch(url, { signal });
            if (!response.ok) throw new Error(`Erreur HTTP: ${response.status}`);
            const resData = await response.json();
            setCompareData(resData);
        } catch (err) {
            if (err.name !== 'AbortError') {
                setError(err.message || 'Impossible de charger la comparaison');
            }
        } finally {
            if (!signal.aborted) setLoading(false);
        }
    };

    useEffect(() => {
        fetchVersions();
    }, []);

    useEffect(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        const controller = new AbortController();
        abortControllerRef.current = controller;

        if (viewMode === 'single') {
            if (selectedVersion) {
                fetchData(selectedVersion, selectedAttraction, controller.signal);
            }
        } else {
            if (v1Version && v2Version) {
                fetchComparison(v1Version, v2Version, selectedAttraction, controller.signal);
            }
        }

        return () => {
            controller.abort();
        };
    }, [viewMode, selectedVersion, selectedAttraction, v1Version, v2Version]);

    // Données pour le Donut Chart quand "ALL" est sélectionné
    const donutAttractionData = useMemo(() => {
        if (!data) return [];
        return data.repartition_par_attraction || [];
    }, [data]);

    // Données filtrées pour le diagramme d'utilisation mémoire
    const rawMemoire = useMemo(() => data?.utilisation_memoire || [], [data]);
    const memoireData = useMemo(() => showAllPie ? rawMemoire : rawMemoire.slice(0, 8), [rawMemoire, showAllPie]);

    // Données filtrées pour le diagramme d'anomalies
    const rawComparaison = useMemo(() => data?.comparaison_colonnes || [], [data]);
    const comparaisonData = useMemo(() => showAllBar ? rawComparaison : rawComparaison.slice(0, 10), [rawComparaison, showAllBar]);

    const statistiquesColonnes = data?.comparaison_colonnes || [];

    // Recherche et Tri du tableau
    const processedTableData = useMemo(() => {
        if (!data || !data.apercu_donnees) return [];
        let rows = [...data.apercu_donnees];

        if (searchTerm.trim() !== '') {
            const keyword = searchTerm.toLowerCase();
            rows = rows.filter(row =>
                Object.values(row).some(val => val !== null && String(val).toLowerCase().includes(keyword))
            );
        }

        if (sortConfig.key !== null) {
            rows.sort((a, b) => {
                let aVal = a[sortConfig.key] ?? '';
                let bVal = b[sortConfig.key] ?? '';

                if (!isNaN(aVal) && !isNaN(bVal) && aVal !== '' && bVal !== '') {
                    aVal = Number(aVal);
                    bVal = Number(bVal);
                } else {
                    aVal = String(aVal).toLowerCase();
                    bVal = String(bVal).toLowerCase();
                }

                if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
                if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
                return 0;
            });
        }
        return rows;
    }, [data, searchTerm, sortConfig]);

    const totalPages = Math.max(1, Math.ceil(processedTableData.length / rowsPerPage));

    const paginatedTableData = useMemo(() => {
        const start = (currentPage - 1) * rowsPerPage;
        return processedTableData.slice(start, start + rowsPerPage);
    }, [processedTableData, currentPage, rowsPerPage]);

    const handleSort = (columnKey) => {
        let direction = 'asc';
        if (sortConfig.key === columnKey && sortConfig.direction === 'asc') direction = 'desc';
        setSortConfig({ key: columnKey, direction });
    };

    const styles = {
        wrapper: { minHeight: '100vh', backgroundColor: '#f8fafc', padding: '24px', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1e293b' },
        container: { maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' },
        card: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)', boxSizing: 'border-box' },
        headerCard: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '16px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' },
        topBar: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' },
        selectBox: { display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '8px 14px', borderRadius: '16px', fontSize: '12px', minWidth: '180px' },
        select: { border: 'none', background: 'transparent', fontWeight: 'bold', color: '#0f172a', outline: 'none', cursor: 'pointer', appearance: 'none', width: '100%', lineHeight: '1' },
        kpiGrid: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '12px 16px', display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' },
        kpiItem: { display: 'flex', alignItems: 'center', gap: '12px', padding: '8px 4px', minWidth: 0 },
        iconBg: (bg, color) => ({ backgroundColor: bg, color: color, padding: '10px', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }),
        tabBtn: (isActive) => ({ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', borderRadius: '14px', fontSize: '13px', fontWeight: 'bold', border: '1px solid', borderColor: isActive ? '#2563eb' : '#e2e8f0', cursor: 'pointer', transition: 'all 0.2s', backgroundColor: isActive ? '#2563eb' : '#ffffff', color: isActive ? '#ffffff' : '#64748b' }),
        actionBtn: { display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', borderRadius: '12px', fontSize: '12px', fontWeight: 'bold', backgroundColor: '#f1f5f9', color: '#475569', border: 'none', cursor: 'pointer' },
        modeToggleBtn: (isActive) => ({ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 14px', borderRadius: '12px', fontSize: '12px', fontWeight: 'bold', border: 'none', cursor: 'pointer', backgroundColor: isActive ? '#0f172a' : '#f1f5f9', color: isActive ? '#ffffff' : '#64748b' }),
        badgeDelta: (val) => {
            const isGood = val <= 0;
            return {
                padding: '2px 8px',
                borderRadius: '8px',
                fontSize: '11px',
                fontWeight: 'bold',
                backgroundColor: isGood ? '#dcfce7' : '#fee2e2',
                color: isGood ? '#15803d' : '#b91c1c'
            };
        },
        pageBtn: (disabled) => ({
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '6px 10px', borderRadius: '8px', border: '1px solid #e2e8f0', backgroundColor: disabled ? '#f1f5f9' : '#ffffff', color: disabled ? '#cbd5e1' : '#475569', cursor: disabled ? 'not-allowed' : 'pointer', fontSize: '12px'
        })
    };

    if (loading && !data && !compareData) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '500px', gap: '16px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '50%', border: '4px solid #dbeafe', borderTopColor: '#2563eb', animation: 'spin 1s linear infinite' }} />
                <p style={{ color: '#64748b', fontWeight: '500', fontSize: '14px' }}>Chargement de l'analyse du schéma data_prep...</p>
                <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
            </div>
        );
    }

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* EN-TÊTE & FILTRES */}
                <div style={styles.headerCard}>
                    <div style={styles.topBar}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={styles.iconBg('#eff6ff', '#2563eb')}>
                                <BarChart3 size={24} />
                            </div>
                            <div>
                                <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2' }}>
                                    Analyse des données
                                </h1>
                                <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8', fontWeight: '600', lineHeight: '1.2' }}>
                                    Profilage et comparaison des données du schéma data_prep
                                </p>
                            </div>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                            <div style={{ display: 'flex', backgroundColor: '#f1f5f9', padding: '4px', borderRadius: '14px', gap: '4px' }}>
                                <button onClick={() => setViewMode('single')} style={styles.modeToggleBtn(viewMode === 'single')}>
                                    <Activity size={14} /> Vue Simple
                                </button>
                                <button onClick={() => setViewMode('compare')} style={styles.modeToggleBtn(viewMode === 'compare')}>
                                    <GitCompare size={14} /> Comparaison
                                </button>
                            </div>

                            {/* Filtre Attraction */}
                            <div style={styles.selectBox}>
                                <Filter size={14} color="#94a3b8" />
                                <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap' }}>Attraction :</span>
                                <select value={selectedAttraction} onChange={(e) => handleAttractionChange(e.target.value)} style={styles.select}>
                                    <option value="ALL">Toutes les attractions</option>
                                    {attractionList.map((id) => (
                                        <option key={id} value={id}>Attraction {id}</option>
                                    ))}
                                </select>
                                <ChevronDown size={14} color="#94a3b8" />
                            </div>
                        </div>
                    </div>

                    {/* STEPPER VERSIONS (Intégration dynamique des couleurs avec getDynamicColor) */}
                    <div style={{
                        backgroundColor: '#ffffff',
                        borderRadius: '16px',
                        border: '1px solid #e2e8f0',
                        padding: '20px 24px',
                        width: '100%',
                        boxSizing: 'border-box'
                    }}>
                        {viewMode === 'single' ? (
                            <div>
                                <div style={{ fontSize: '14px', fontWeight: '700', color: '#1e293b', marginBottom: '20px' }}>
                                    Choisissez la version des données
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between', overflowX: 'auto' }}>
                                    {filteredVersionList.map((ver, index) => {
                                        const isSelected = selectedVersion === ver;
                                        const isLast = index === filteredVersionList.length - 1;
                                        const totalVer = filteredVersionList.length;
                                        const dynamicColor = getDynamicColor(index, totalVer);

                                        return (
                                            <React.Fragment key={ver}>
                                                <div
                                                    onClick={() => setSelectedVersion(ver)}
                                                    style={{
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        gap: '8px',
                                                        cursor: 'pointer',
                                                        userSelect: 'none',
                                                        flexShrink: 0
                                                    }}
                                                >
                                                    <div style={{
                                                        width: '28px',
                                                        height: '28px',
                                                        borderRadius: '50%',
                                                        backgroundColor: isSelected ? dynamicColor : '#94a3b8',
                                                        color: '#ffffff',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'center',
                                                        fontSize: '12px',
                                                        fontWeight: '700',
                                                        transition: 'all 0.2s ease',
                                                        boxShadow: isSelected ? `0 0 0 4px ${dynamicColor}25` : 'none',
                                                        flexShrink: 0
                                                    }}>
                                                        {index + 1}
                                                    </div>
                                                    <span style={{
                                                        fontSize: '13px',
                                                        fontWeight: isSelected ? '700' : '500',
                                                        color: isSelected ? '#1e293b' : '#64748b',
                                                        whiteSpace: 'nowrap'
                                                    }}>
                                                        {ver}
                                                    </span>
                                                </div>

                                                {!isLast && (
                                                    <div style={{
                                                        flex: 1,
                                                        height: '1px',
                                                        backgroundColor: '#e2e8f0',
                                                        margin: '0 12px',
                                                        minWidth: '16px'
                                                    }} />
                                                )}
                                            </React.Fragment>
                                        );
                                    })}
                                </div>
                            </div>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                <div>
                                    <div style={{ fontSize: '14px', fontWeight: '700', color: '#1e293b', marginBottom: '20px' }}>
                                        Choisissez la version initiale
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between', overflowX: 'auto' }}>
                                        {filteredVersionList.map((ver, index) => {
                                            const isSelected = v1Version === ver;
                                            const isLast = index === filteredVersionList.length - 1;
                                            const totalVer = filteredVersionList.length;
                                            const dynamicColor = getDynamicColor(index, totalVer);

                                            return (
                                                <React.Fragment key={`v1-${ver}`}>
                                                    <div
                                                        onClick={() => setV1Version(ver)}
                                                        style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', userSelect: 'none', flexShrink: 0 }}
                                                    >
                                                        <div style={{
                                                            width: '28px', height: '28px', borderRadius: '50%',
                                                            backgroundColor: isSelected ? dynamicColor : '#94a3b8',
                                                            color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                            fontSize: '12px', fontWeight: '700', flexShrink: 0,
                                                            boxShadow: isSelected ? `0 0 0 4px ${dynamicColor}25` : 'none'
                                                        }}>
                                                            {index + 1}
                                                        </div>
                                                        <span style={{ fontSize: '13px', fontWeight: isSelected ? '700' : '500', color: isSelected ? '#1e293b' : '#64748b', whiteSpace: 'nowrap' }}>
                                                            {ver}
                                                        </span>
                                                    </div>
                                                    {!isLast && <div style={{ flex: 1, height: '1px', backgroundColor: '#e2e8f0', margin: '0 12px', minWidth: '16px' }} />}
                                                </React.Fragment>
                                            );
                                        })}
                                    </div>
                                </div>

                                <div>
                                    <div style={{ fontSize: '14px', fontWeight: '700', color: '#1e293b', marginBottom: '20px' }}>
                                        Choisissez la version mise à jour
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between', overflowX: 'auto' }}>
                                        {filteredVersionList.map((ver, index) => {
                                            const isSelected = v2Version === ver;
                                            const isLast = index === filteredVersionList.length - 1;
                                            const totalVer = filteredVersionList.length;
                                            const dynamicColor = getDynamicColor(index, totalVer);

                                            return (
                                                <React.Fragment key={`v2-${ver}`}>
                                                    <div
                                                        onClick={() => setV2Version(ver)}
                                                        style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', userSelect: 'none', flexShrink: 0 }}
                                                    >
                                                        <div style={{
                                                            width: '28px', height: '28px', borderRadius: '50%',
                                                            backgroundColor: isSelected ? dynamicColor : '#94a3b8',
                                                            color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                            fontSize: '12px', fontWeight: '700', flexShrink: 0,
                                                            boxShadow: isSelected ? `0 0 0 4px ${dynamicColor}25` : 'none'
                                                        }}>
                                                            {index + 1}
                                                        </div>
                                                        <span style={{ fontSize: '13px', fontWeight: isSelected ? '700' : '500', color: isSelected ? '#1e293b' : '#64748b', whiteSpace: 'nowrap' }}>
                                                            {ver}
                                                        </span>
                                                    </div>
                                                    {!isLast && <div style={{ flex: 1, height: '1px', backgroundColor: '#e2e8f0', margin: '0 12px', minWidth: '16px' }} />}
                                                </React.Fragment>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {error && (
                    <div style={{ padding: '16px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '16px', color: '#991b1b', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <AlertTriangle size={20} color="#dc2626" />
                        <span>{error}</span>
                    </div>
                )}

                {/* VUE SIMPLE */}
                {viewMode === 'single' && data && (
                    <>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '16px' }}>
                            <div style={{ ...styles.card, display: 'flex', alignItems: 'center', gap: '16px', padding: '16px 20px' }}>
                                <div style={styles.iconBg('#eff6ff', '#2563eb')}><Database size={22} /></div>
                                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                                    <div style={{ fontSize: '11px', fontWeight: '700', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Total Lignes</div>
                                    <div style={{ fontSize: '20px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2', marginTop: '4px' }}>{(data.total_lignes || 0).toLocaleString()}</div>
                                </div>
                            </div>

                            <div style={{ ...styles.card, display: 'flex', alignItems: 'center', gap: '16px', padding: '16px 20px' }}>
                                <div style={styles.iconBg('#faf5ff', '#9333ea')}><Layers size={22} /></div>
                                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                                    <div style={{ fontSize: '11px', fontWeight: '700', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Total Variables</div>
                                    <div style={{ fontSize: '20px', fontWeight: '800', color: '#0f172a', lineHeight: '1.2', marginTop: '4px' }}>{(data.total_colonnes || 0).toLocaleString()}</div>
                                </div>
                            </div>
                        </div>

                        <div style={styles.kpiGrid}>
                            <div style={styles.kpiItem}>
                                <div style={styles.iconBg('#fffbeb', '#d97706')}><AlertTriangle size={18} /></div>
                                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                                    <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Manquantes</div>
                                    <div style={{ fontSize: '14px', fontWeight: '800', color: '#d97706', lineHeight: '1.2', marginTop: '2px' }}>
                                        {(data.nb_valeurs_manquantes || 0).toLocaleString()} <span style={{ fontSize: '11px' }}>({data.pourcentage_manquants || 0}%)</span>
                                    </div>
                                </div>
                            </div>

                            <div style={styles.kpiItem}>
                                <div style={styles.iconBg('#fef2f2', '#dc2626')}><Activity size={18} /></div>
                                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                                    <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Outliers</div>
                                    <div style={{ fontSize: '14px', fontWeight: '800', color: '#dc2626', lineHeight: '1.2', marginTop: '2px' }}>
                                        {(data.nb_outliers || 0).toLocaleString()} <span style={{ fontSize: '11px' }}>({data.pourcentage_outliers || 0}%)</span>
                                    </div>
                                </div>
                            </div>

                            <div style={styles.kpiItem}>
                                <div style={styles.iconBg('#fff1f2', '#e11d48')}><TrendingDown size={18} /></div>
                                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                                    <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Négatives</div>
                                    <div style={{ fontSize: '14px', fontWeight: '800', color: '#e11d48', lineHeight: '1.2', marginTop: '2px' }}>
                                        {(data.nb_valeurs_negatives || 0).toLocaleString()} <span style={{ fontSize: '11px' }}>({data.pourcentage_negatifs || 0}%)</span>
                                    </div>
                                </div>
                            </div>

                            <div style={styles.kpiItem}>
                                <div style={styles.iconBg('#faf5ff', '#9333ea')}><Copy size={18} /></div>
                                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                                    <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', lineHeight: '1.2' }}>Doublons</div>
                                    <div style={{ fontSize: '14px', fontWeight: '800', color: '#9333ea', lineHeight: '1.2', marginTop: '2px' }}>
                                        {(data.nb_doublons || 0).toLocaleString()} <span style={{ fontSize: '11px' }}>({data.pourcentage_dupliques || 0}%)</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* SECTION GRAPHIQUES */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '20px' }}>

                            <div style={styles.card}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px', alignItems: 'center' }}>
                                    <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px', lineHeight: '1.2' }}>
                                        <PieIcon size={18} color="#2563eb" />
                                        {selectedAttraction === 'ALL'
                                            ? `Répartition par Attraction (${selectedVersion})`
                                            : `Répartition du Volume de ${selectedAttraction} par Version`}
                                    </h2>
                                </div>

                                <div style={{ width: '100%', height: '280px' }}>
                                    {selectedAttraction === 'ALL' ? (
                                        donutAttractionData.length > 0 ? (
                                            <ResponsiveContainer width="100%" height="100%">
                                                <PieChart>
                                                    <Pie
                                                        data={donutAttractionData}
                                                        dataKey="pourcentage"
                                                        nameKey="id_attraction"
                                                        cx="50%"
                                                        cy="50%"
                                                        outerRadius={85}
                                                        innerRadius={45}
                                                        paddingAngle={4}
                                                    >
                                                        {donutAttractionData.map((e, idx) => (
                                                            <Cell key={idx} fill={getDynamicColor(idx, donutAttractionData.length)} />
                                                        ))}
                                                    </Pie>
                                                    <Tooltip formatter={(val, name) => [`${val}%`, `Attraction: ${name}`]} />
                                                </PieChart>
                                            </ResponsiveContainer>
                                        ) : (
                                            <div style={{ padding: '40px', textAlign: 'center', color: '#94A3B8', fontSize: '13px' }}>
                                                Données non disponibles
                                            </div>
                                        )
                                    ) : (
                                        data.repartition_par_version && data.repartition_par_version.length > 0 ? (
                                            <ResponsiveContainer width="100%" height="100%">
                                                <PieChart>
                                                    <Pie
                                                        data={data.repartition_par_version}
                                                        dataKey="total_lignes"
                                                        nameKey="name"
                                                        cx="50%"
                                                        cy="50%"
                                                        outerRadius={85}
                                                        innerRadius={45}
                                                        paddingAngle={4}
                                                    >
                                                        {data.repartition_par_version.map((entry, idx) => (
                                                            <Cell key={idx} fill={getDynamicColor(idx, data.repartition_par_version.length)} />
                                                        ))}
                                                    </Pie>
                                                    <Tooltip
                                                        formatter={(value, name, props) => {
                                                            const total = data.repartition_par_version.reduce((acc, item) => acc + item.total_lignes, 0);
                                                            const percent = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                                            return [`${value.toLocaleString()} lignes (${percent}%)`, `Version: ${name}`];
                                                        }}
                                                    />
                                                </PieChart>
                                            </ResponsiveContainer>
                                        ) : (
                                            <div style={{ padding: '40px', textAlign: 'center', color: '#94A3B8', fontSize: '13px' }}>
                                                Aucune version trouvée pour cette attraction
                                            </div>
                                        )
                                    )}
                                </div>
                            </div>

                            <div style={styles.card}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px', alignItems: 'center' }}>
                                    <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', lineHeight: '1.2' }}>Utilisation Mémoire par Colonne</h2>
                                    {rawMemoire.length > 8 && (
                                        <button onClick={() => setShowAllPie(!showAllPie)} style={styles.actionBtn}>
                                            {showAllPie ? <EyeOff size={14} /> : <Eye size={14} />}
                                            {showAllPie ? "Top 8" : `Voir tout (${rawMemoire.length})`}
                                        </button>
                                    )}
                                </div>
                                <div style={{ width: '100%', height: '280px' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie data={memoireData} dataKey="pourcentage" nameKey="nom_colonne" cx="50%" cy="50%" outerRadius={85} innerRadius={45} paddingAngle={3}>
                                                {memoireData.map((e, idx) => <Cell key={idx} fill={getDynamicColor(idx, memoireData.length)} />)}
                                            </Pie>
                                            <Tooltip formatter={(value, name) => [`${value}%`, `Colonne: ${name}`]} />
                                        </PieChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>

                        {/* BAR CHART: Anomalies par Colonne */}
                        <div style={styles.card}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px', alignItems: 'center' }}>
                                <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', lineHeight: '1.2' }}>Anomalies par Colonne (Manquants & Outliers)</h2>
                                {rawComparaison.length > 10 && (
                                    <button onClick={() => setShowAllBar(!showAllBar)} style={styles.actionBtn}>
                                        {showAllBar ? <EyeOff size={14} /> : <Eye size={14} />}
                                        {showAllBar ? "Top 10" : `Voir tout (${rawComparaison.length})`}
                                    </button>
                                )}
                            </div>
                            <div style={{ width: '100%', height: '280px' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={comparaisonData} margin={{ top: 10, right: 10, left: -20, bottom: 45 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                                        <XAxis dataKey="nom_colonne" angle={-35} textAnchor="end" interval={0} tick={{ fill: '#64748b', fontSize: 10 }} />
                                        <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                                        <Tooltip />
                                        <Bar dataKey="valeurs_manquantes" name="Manquants" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                                        <Bar dataKey="valeurs_aberrantes" name="Outliers" fill="#ef4444" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid #e2e8f0', paddingBottom: '12px' }}>
                            <button onClick={() => setActiveTab('overview')} style={styles.tabBtn(activeTab === 'overview')}>
                                <Activity size={16} /> Statistiques Détaillées
                            </button>
                            <button onClick={() => setActiveTab('preview')} style={styles.tabBtn(activeTab === 'preview')}>
                                <Table size={16} /> Aperçu Interactif
                            </button>
                        </div>

                        {activeTab === 'overview' && (
                            <div style={styles.card}>
                                <div style={{ overflowX: 'auto', borderRadius: '16px', border: '1px solid #e2e8f0' }}>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px' }}>
                                        <thead>
                                            <tr style={{ backgroundColor: '#f8fafc', color: '#475569', fontWeight: 'bold', textTransform: 'uppercase' }}>
                                                <th style={{ padding: '12px 16px' }}>Colonne</th>
                                                <th style={{ padding: '12px 16px' }}>Type</th>
                                                <th style={{ padding: '12px 16px', textAlign: 'right' }}>Manquants</th>
                                                <th style={{ padding: '12px 16px', textAlign: 'right' }}>Outliers</th>
                                                <th style={{ padding: '12px 16px', textAlign: 'right' }}>Moyenne</th>
                                                <th style={{ padding: '12px 16px', textAlign: 'right' }}>Écart-type</th>
                                                <th style={{ padding: '12px 16px', textAlign: 'right' }}>Min</th>
                                                <th style={{ padding: '12px 16px', textAlign: 'right' }}>Médiane</th>
                                                <th style={{ padding: '12px 16px', textAlign: 'right' }}>Max</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {statistiquesColonnes.map((col, idx) => (
                                                <tr key={idx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                    <td style={{ padding: '12px 16px', fontWeight: 'bold' }}>{col.nom_colonne}</td>
                                                    <td style={{ padding: '12px 16px' }}><span style={{ backgroundColor: '#f1f5f9', padding: '2px 6px', borderRadius: '6px' }}>{col.type_donnees}</span></td>
                                                    <td style={{ padding: '12px 16px', textAlign: 'right', color: '#d97706', fontWeight: 'bold' }}>{col.valeurs_manquantes}</td>
                                                    <td style={{ padding: '12px 16px', textAlign: 'right', color: '#dc2626', fontWeight: 'bold' }}>{col.valeurs_aberrantes}</td>
                                                    <td style={{ padding: '12px 16px', textAlign: 'right' }}>{col.moyenne ?? '-'}</td>
                                                    <td style={{ padding: '12px 16px', textAlign: 'right' }}>{col.ecart_type ?? '-'}</td>
                                                    <td style={{ padding: '12px 16px', textAlign: 'right' }}>{col.val_min ?? '-'}</td>
                                                    <td style={{ padding: '12px 16px', textAlign: 'right' }}>{col.mediane ?? '-'}</td>
                                                    <td style={{ padding: '12px 16px', textAlign: 'right' }}>{col.val_max ?? '-'}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}

                        {activeTab === 'preview' && (
                            <div style={{ ...styles.card, display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', lineHeight: '1.2' }}>Aperçu des Données ({selectedVersion})</h2>
                                    <div style={{ display: 'flex', gap: '12px' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', padding: '6px 12px', borderRadius: '12px', width: '200px' }}>
                                            <Search size={14} color="#94a3b8" />
                                            <input type="text" placeholder="Rechercher..." value={searchTerm} onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }} style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', fontSize: '12px', lineHeight: '1' }} />
                                        </div>
                                    </div>
                                </div>

                                <div style={{ width: '100%', maxHeight: '450px', overflow: 'auto', border: '1px solid #e2e8f0', borderRadius: '16px' }}>
                                    <table style={{ width: 'max-content', minWidth: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                                        <thead>
                                            <tr style={{ backgroundColor: '#f8fafc', color: '#475569', fontWeight: 'bold', position: 'sticky', top: 0 }}>
                                                {data.apercu_donnees?.length > 0 && Object.keys(data.apercu_donnees[0]).map((col) => (
                                                    <th key={col} onClick={() => handleSort(col)} style={{ padding: '12px 16px', cursor: 'pointer', borderBottom: '1px solid #e2e8f0' }}>
                                                        {col} {sortConfig.key === col ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {paginatedTableData.map((row, idx) => (
                                                <tr key={idx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                    {Object.values(row).map((val, cIdx) => (
                                                        <td key={cIdx} style={{ padding: '10px 16px', fontFamily: 'monospace' }}>
                                                            {val !== null && val !== "" ? String(val) : <span style={{ color: '#cbd5e1', fontStyle: 'italic' }}>null</span>}
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>

                                {/* PAGINATION */}
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '8px', fontSize: '12px', color: '#64748b' }}>
                                    <div>
                                        Affichage de {processedTableData.length === 0 ? 0 : (currentPage - 1) * rowsPerPage + 1} à {Math.min(currentPage * rowsPerPage, processedTableData.length)} sur {processedTableData.length} lignes
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <select
                                            value={rowsPerPage}
                                            onChange={(e) => { setRowsPerPage(Number(e.target.value)); setCurrentPage(1); }}
                                            style={{ padding: '4px 8px', borderRadius: '8px', border: '1px solid #e2e8f0', backgroundColor: '#ffffff', fontSize: '12px', color: '#475569', lineHeight: '1' }}
                                        >
                                            <option value={10}>10 / page</option>
                                            <option value={25}>25 / page</option>
                                            <option value={50}>50 / page</option>
                                        </select>
                                        <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                                            <button
                                                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                                                disabled={currentPage === 1}
                                                style={styles.pageBtn(currentPage === 1)}
                                            >
                                                <ChevronLeft size={16} />
                                            </button>
                                            <span style={{ padding: '6px 12px', fontWeight: 'bold', color: '#0f172a', lineHeight: '1' }}>
                                                {currentPage} / {totalPages}
                                            </span>
                                            <button
                                                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                                                disabled={currentPage === totalPages}
                                                style={styles.pageBtn(currentPage === totalPages)}
                                            >
                                                <ChevronRight size={16} />
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </>
                )}

                {/* VUE COMPARAISON */}
                {viewMode === 'compare' && compareData && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        <div style={styles.card}>
                            <h2 style={{ margin: '0 0 16px 0', fontSize: '15px', fontWeight: 'bold', lineHeight: '1.2' }}>
                                Évolution Globale ({compareData.v1_version_id} ➔ {compareData.v2_version_id})
                            </h2>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '16px' }}>
                                <div style={{ padding: '16px', border: '1px solid #e2e8f0', borderRadius: '16px', backgroundColor: '#f8fafc' }}>
                                    <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 'bold', lineHeight: '1.2' }}>VALEURS MANQUANTES</div>
                                    <div style={{ fontSize: '18px', fontWeight: '800', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '8px', lineHeight: '1.2' }}>
                                        <span>{compareData.delta_global?.diff_nb_valeurs_manquantes ?? 0}</span>
                                        <span style={styles.badgeDelta(compareData.delta_global?.diff_nb_valeurs_manquantes ?? 0)}>
                                            {compareData.delta_global?.diff_pourcentage_manquants ?? 0}%
                                        </span>
                                    </div>
                                </div>

                                <div style={{ padding: '16px', border: '1px solid #e2e8f0', borderRadius: '16px', backgroundColor: '#f8fafc' }}>
                                    <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 'bold', lineHeight: '1.2' }}>VALEURS ABERRANTES (OUTLIERS)</div>
                                    <div style={{ fontSize: '18px', fontWeight: '800', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '8px', lineHeight: '1.2' }}>
                                        <span>{compareData.delta_global?.diff_nb_outliers ?? 0}</span>
                                        <span style={styles.badgeDelta(compareData.delta_global?.diff_nb_outliers ?? 0)}>
                                            {compareData.delta_global?.diff_pourcentage_outliers ?? 0}%
                                        </span>
                                    </div>
                                </div>

                                <div style={{ padding: '16px', border: '1px solid #e2e8f0', borderRadius: '16px', backgroundColor: '#f8fafc' }}>
                                    <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 'bold', lineHeight: '1.2' }}>VARIATION DES LIGNES</div>
                                    <div style={{ fontSize: '18px', fontWeight: '800', marginTop: '4px', lineHeight: '1.2' }}>
                                        {compareData.delta_global?.diff_total_lignes ?? 0}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div style={styles.card}>
                            <h2 style={{ margin: '0 0 16px 0', fontSize: '15px', fontWeight: 'bold', lineHeight: '1.2' }}>
                                Comparaison Détaillée des Variables
                            </h2>
                            <div style={{ overflowX: 'auto', borderRadius: '16px', border: '1px solid #e2e8f0' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px' }}>
                                    <thead>
                                        <tr style={{ backgroundColor: '#f8fafc', color: '#475569', fontWeight: 'bold', textTransform: 'uppercase' }}>
                                            <th style={{ padding: '12px 16px' }}>Colonne</th>
                                            <th style={{ padding: '12px 16px' }}>Type (V1 ➔ V2)</th>
                                            <th style={{ padding: '12px 16px', textAlign: 'right' }}>Δ Manquants</th>
                                            <th style={{ padding: '12px 16px', textAlign: 'right' }}>Δ Outliers</th>
                                            <th style={{ padding: '12px 16px', textAlign: 'right' }}>Δ Moyenne</th>
                                            <th style={{ padding: '12px 16px', textAlign: 'right' }}>Δ Médiane</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {(compareData.delta_colonnes || []).map((col, idx) => (
                                            <tr key={idx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                <td style={{ padding: '12px 16px', fontWeight: 'bold' }}>{col.nom_colonne}</td>
                                                <td style={{ padding: '12px 16px' }}>
                                                    <span style={{ backgroundColor: '#f1f5f9', padding: '2px 6px', borderRadius: '6px' }}>
                                                        {col.type_v1} ➔ {col.type_v2}
                                                    </span>
                                                </td>
                                                <td style={{ padding: '12px 16px', textAlign: 'right' }}>
                                                    <span style={styles.badgeDelta(col.diff_valeurs_manquantes)}>
                                                        {col.diff_valeurs_manquantes > 0 ? `+${col.diff_valeurs_manquantes}` : col.diff_valeurs_manquantes}
                                                    </span>
                                                </td>
                                                <td style={{ padding: '12px 16px', textAlign: 'right' }}>
                                                    <span style={styles.badgeDelta(col.diff_valeurs_aberrantes)}>
                                                        {col.diff_valeurs_aberrantes > 0 ? `+${col.diff_valeurs_aberrantes}` : col.diff_valeurs_aberrantes}
                                                    </span>
                                                </td>
                                                <td style={{ padding: '12px 16px', textAlign: 'right', fontWeight: 'bold' }}>
                                                    {col.diff_moyenne !== null && col.diff_moyenne !== undefined ? (col.diff_moyenne > 0 ? `+${col.diff_moyenne}` : col.diff_moyenne) : '-'}
                                                </td>
                                                <td style={{ padding: '12px 16px', textAlign: 'right', fontWeight: 'bold' }}>
                                                    {col.diff_mediane !== null && col.diff_mediane !== undefined ? (col.diff_mediane > 0 ? `+${col.diff_mediane}` : col.diff_mediane) : '-'}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}