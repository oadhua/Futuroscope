import React, { useState, useEffect } from 'react';
import {
    Database, AlertTriangle, Layers, Filter,
    ChevronDown, Table, CheckCircle2, UserCheck,
    RefreshCw, Info, Sliders
} from 'lucide-react';

// Từ điển định nghĩa mô tả và đơn vị bằng tiếng Pháp
const COLUMN_METADATA_MAP = {
    // --- DIM_ATTRACTION & DIM_CADENCE ---
    id_attraction: { description: "Identifiant unique du bâtiment ou de l'attraction du parc.", owner: "Exploitation Futuroscope" },
    name: { description: "Nom officiel de l'attraction ou de l'équipement.", owner: "Exploitation Futuroscope" },
    surface: { description: "Superficie totale de l'attraction ou du bâtiment (m²).", owner: "Futuroscope Maintenance et Développement" },
    capacite_salle: { description: "Capacité maximale d'accueil de la salle principale (personnes).", owner: "Exploitation Futuroscope" },
    capacite_file_attente: { description: "Capacité maximale d'accueil de la file d'attente (personnes).", owner: "Exploitation Futuroscope" },
    capacite_pre_salle: { description: "Capacité maximale d'accueil de la pré-salle (personnes).", owner: "Exploitation Futuroscope" },
    duree_longue: { description: "Durée d'un cycle long de spectacle ou d'animation.", owner: "Exploitation Futuroscope" },
    duree_courte: { description: "Durée d'un cycle court de spectacle ou d'animation.", owner: "Exploitation Futuroscope" },
    cycle_max_vl: { description: "Nombre maximum de cycles par heure en version longue (VL).", owner: "Futuroscope Maintenance et Développement" },
    duty_cycle_max_vl: { description: "Rapport cyclique maximal de fonctionnement en version longue (VL).", owner: "Futuroscope Maintenance et Développement" },
    cycle_max_vc: { description: "Nombre maximum de cycles par heure en version courte (VC).", owner: "Futuroscope Maintenance et Développement" },
    duty_cycle_max_vc: { description: "Rapport cyclique maximal de fonctionnement en version courte (VC).", owner: "Futuroscope Maintenance et Développement" },

    // --- DIM_TEMPS & DIM_HORAIRE ---
    temps_id: { description: "Clé primaire temporelle unique au format horaire (YYYYMMDDHH).", owner: "Système / Entrepôt de Données" },
    datetime: { description: "Horodatage complet de l'enregistrement (YYYY-MM-DD HH:mm:ss).", owner: "Système / Passerelle IoT" },
    date: { description: "Date calendaire au format YYYY-MM-DD.", owner: "Système / Entrepôt de Données" },
    annee: { description: "Année associée à l'enregistrement.", owner: "Système / Entrepôt de Données" },
    mois: { description: "Mois de l'année (1 à 12).", owner: "Système / Entrepôt de Données" },
    jour: { description: "Jour du mois (1 à 31).", owner: "Système / Entrepôt de Données" },
    heure: { description: "Heure de la journée (0 à 23).", owner: "Système / Passerelle IoT" },
    jour_semaine: { description: "Jour de la semaine (1 = Lundi, 7 = Dimanche).", owner: "Système / Entrepôt de Données" },
    is_weekend: { description: "Indicateur binaire de fin de semaine (1 = Oui, 0 = Non).", owner: "Système / Entrepôt de Données" },
    jf: { description: "Indicateur binaire de jour férié (1 = Jour Férié, 0 = Jour Normal).", owner: "Système / Entrepôt de Données" },
    is_open: { description: "Statut d'ouverture globale du parc (1 = Ouvert, 0 = Fermé).", owner: "Exploitation Futuroscope" },
    h_ouv: { description: "Heure d'ouverture officielle du parc.", owner: "Exploitation Futuroscope" },
    h_ferm: { description: "Heure de fermeture officielle du parc.", owner: "Exploitation Futuroscope" },
    frequentation: { description: "Nombre total de visiteurs enregistrés sur la journée au parc.", owner: "Système de Billetterie Futuroscope" },
    type_frequentation: { description: "Catégorie de niveau de fréquentation attendue.", owner: "Exploitation Futuroscope" },

    // --- DIM_WEATHER ---
    date_key: { description: "Clé calendaire pour la jointure météo.", owner: "Météo France / Station Météo" },
    temperature: { description: "Température extérieure ambiante horaire (°C).", owner: "Météo France / Capteurs Météo" },
    humidite: { description: "Taux d'humidité relative dans l'air (%).", owner: "Météo France / Capteurs Météo" },
    rayonnement_solaire: { description: "Niveau de rayonnement solaire global (W/m²).", owner: "Météo France / Capteurs Météo" },
    day_degree_cold: { description: "Degrés-jours de climatisation / refroidissement.", owner: "Futuroscope Maintenance et Développement" },
    day_degree_hot: { description: "Degrés-jours de chauffage (DJU).", owner: "Futuroscope Maintenance et Développement" },

    // --- FACT TABLES ---
    visitor_count: { description: "Nombre de passages / visiteurs enregistrés à l'attraction pendant l'heure.", owner: "Système de Comptage / Billetterie" },
    ouvert: { description: "Durée ou ratio de fonctionnement normal de l'attraction.", owner: "Exploitation Futuroscope" },
    interrompu: { description: "Durée ou ratio d'interruption / panne de l'attraction.", owner: "Futuroscope Maintenance et Développement" },
    operation: { description: "Durée ou ratio d'exploitation technique de l'attraction.", owner: "Futuroscope Maintenance et Développement" },
    conso_elec: { description: "Consommation d'électricité horaire agrégée (kWh).", owner: "Futuroscope Maintenance et Développement" },
    conso_ec: { description: "Consommation d'eau chaude / énergie thermique horaire agrégée.", owner: "Futuroscope Maintenance et Développement" }
};

export default function InitialProfiling() {
    // --- States ---
    const [data, setData] = useState(null);
    const [versions, setVersions] = useState([]);
    const [attractions, setAttractions] = useState([]);
    const [selectedVersion, setSelectedVersion] = useState('v1');
    const [selectedAttraction, setSelectedAttraction] = useState('ALL');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const API_BASE_URL = 'http://localhost:8000/initial-profiling';

    // 1. Fetch danh sách Data Versions
    useEffect(() => {
        const fetchVersions = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/versions`);
                if (res.ok) {
                    const verData = await res.json();
                    setVersions(verData);
                    if (verData.length > 0) {
                        setSelectedVersion(verData[0].version_id);
                    }
                }
            } catch (err) {
                console.warn("Lỗi lấy danh sách versions:", err);
            }
        };
        fetchVersions();
    }, []);

    // 2. Fetch danh sách Attractions
    useEffect(() => {
        const fetchAttractions = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/attractions`);
                if (res.ok) {
                    const attrData = await res.json();
                    setAttractions(attrData);
                }
            } catch (err) {
                console.warn("Lỗi lấy danh sách attractions:", err);
            }
        };
        fetchAttractions();
    }, []);

    // 3. Fetch dữ liệu Initial Profiling
    const fetchProfilingData = async () => {
        setLoading(true);
        setError(null);
        try {
            const queryParams = new URLSearchParams({
                version: selectedVersion,
                ...(selectedAttraction !== 'ALL' && { id_attraction: selectedAttraction }),
            });

            const res = await fetch(`${API_BASE_URL}?${queryParams.toString()}`);
            if (!res.ok) throw new Error(`Erreur HTTP: ${res.status}`);

            const result = await res.json();
            setData(result);
        } catch (err) {
            setError(err.message || "Impossible de charger le profilage initial de la base de données.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (selectedVersion) {
            fetchProfilingData();
        }
    }, [selectedVersion, selectedAttraction]);

    // Metadata Helper
    const getColumnMeta = (colName) => {
        if (!colName) return { description: "Métadonnée non disponible.", owner: "Équipe Technique Futuroscope" };
        const key = colName.toLowerCase();
        return COLUMN_METADATA_MAP[key] || {
            description: `Indicateur mesuré (${colName}) du jeu de données du Futuroscope.`,
            owner: "Futuroscope Maintenance et Développement"
        };
    };

    // Render Type Badge
    const renderTypeBadge = (type) => {
        const typeStyles = {
            Integer: { backgroundColor: '#dbeafe', color: '#1e40af' },
            Float: { backgroundColor: '#d1fae5', color: '#065f46' },
            Date: { backgroundColor: '#fef3c7', color: '#92400e' },
            Boolean: { backgroundColor: '#f3e8ff', color: '#6b21a8' },
            String: { backgroundColor: '#f1f5f9', color: '#334155' },
        };
        const currentStyle = typeStyles[type] || typeStyles.String;

        return (
            <span style={{
                padding: '4px 10px',
                borderRadius: '12px',
                fontSize: '11px',
                fontWeight: 'bold',
                display: 'inline-block',
                ...currentStyle
            }}>
                {type}
            </span>
        );
    };

    // --- Đồng bộ 100% Hệ thống Unified Internal Styles từ SingleColumn ---
    const styles = {
        wrapper: { minHeight: '100vh', backgroundColor: '#f8fafc', padding: '24px', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1e293b' },
        container: { maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' },
        card: { backgroundColor: '#ffffff', borderRadius: '24px', border: '1px solid #e2e8f0', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)', boxSizing: 'border-box' },
        headerCard: {
            backgroundColor: '#ffffff',
            borderRadius: '24px',
            border: '1px solid #e2e8f0',
            padding: '16px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '16px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
            minHeight: '80px',
            boxSizing: 'border-box'
        },
        selectBox: {
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            backgroundColor: '#f8fafc',
            border: '1px solid #e2e8f0',
            padding: '8px 14px',
            borderRadius: '16px',
            fontSize: '12px',
            minWidth: '220px',
            maxWidth: '280px',
            flexShrink: 0
        },
        select: {
            border: 'none',
            background: 'transparent',
            fontWeight: 'bold',
            color: '#0f172a',
            outline: 'none',
            cursor: 'pointer',
            appearance: 'none',
            width: '100%',
            textOverflow: 'ellipsis',
            overflow: 'hidden',
            whiteSpace: 'nowrap'
        },
        kpiGrid: {
            backgroundColor: '#ffffff',
            borderRadius: '24px',
            border: '1px solid #e2e8f0',
            padding: '12px 16px',
            display: 'grid',
            gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
            gap: '12px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
            minHeight: '80px',
            boxSizing: 'border-box'
        },
        kpiItem: { display: 'flex', alignItems: 'center', gap: '12px', padding: '8px 4px', minWidth: 0 },
        iconBg: (bg, color) => ({ backgroundColor: bg, color: color, padding: '10px', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }),
        table: { width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' },
        th: { padding: '12px 16px', fontSize: '11px', fontWeight: 'bold', textTransform: 'uppercase', color: '#64748b', borderBottom: '1px solid #e2e8f0', letterSpacing: '0.05em' },
        td: { padding: '14px 16px', borderBottom: '1px solid #f1f5f9', verticalAlign: 'middle' },
    };

    if (loading) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '500px', gap: '16px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '50%', border: '4px solid #dbeafe', borderTopColor: '#2563eb', animation: 'spin 1s linear infinite' }} />
                <p style={{ color: '#64748b', fontWeight: '500', fontSize: '14px' }}>Analyse du schéma et profilage initial en cours...</p>
                <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div style={{ maxWidth: '500px', margin: '48px auto', padding: '24px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '20px', display: 'flex', alignItems: 'center', gap: '16px', color: '#991b1b' }}>
                <AlertTriangle size={24} color="#dc2626" />
                <div>
                    <h3 style={{ margin: 0, fontWeight: 'bold', fontSize: '15px' }}>Erreur de chargement</h3>
                    <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: '#b91c1c' }}>{error || "Aucune donnée retournée par le serveur."}</p>
                </div>
            </div>
        );
    }

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>

                {/* 1. TOP HEADER & FILTER CARD */}
                <div style={styles.headerCard}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flex: '1 1 auto', minWidth: 0 }}>
                        <div style={styles.iconBg('#eff6ff', '#2563eb')}>
                            <Database size={24} />
                        </div>
                        <div style={{ minWidth: 0, overflow: 'hidden' }}>
                            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.02em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                Profilage Initial des Données
                            </h1>
                        </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0 }}>
                        {/* Dropdown Version */}
                        <div style={styles.selectBox}>
                            <Layers size={15} color="#94a3b8" style={{ flexShrink: 0 }} />
                            <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>Version :</span>
                            <select value={selectedVersion} onChange={(e) => setSelectedVersion(e.target.value)} style={styles.select}>
                                {versions.length > 0 ? (
                                    versions.map(v => (
                                        <option key={v.version_id} value={v.version_id}>
                                            {v.version_label} {v.is_updated ? '(Mis à jour)' : ''}
                                        </option>
                                    ))
                                ) : (
                                    <option value="v1">Version 1</option>
                                )}
                            </select>
                            <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                        </div>

                        {/* Dropdown Attraction */}
                        <div style={styles.selectBox}>
                            <Filter size={15} color="#94a3b8" style={{ flexShrink: 0 }} />
                            <span style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap', flexShrink: 0 }}>Attraction :</span>
                            <select value={selectedAttraction} onChange={(e) => setSelectedAttraction(e.target.value)} style={styles.select}>
                                <option value="ALL">Toutes les attractions</option>
                                {attractions.map(id => <option key={id} value={id}>Attraction {id}</option>)}
                            </select>
                            <ChevronDown size={14} color="#94a3b8" style={{ flexShrink: 0 }} />
                        </div>
                    </div>
                </div>

                {/* 2. KPI METRICS BAR */}
                <div style={styles.kpiGrid}>
                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#eff6ff', '#2563eb')}><Table size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Total Lignes</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#0f172a', marginTop: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {data.total_rows ? data.total_rows.toLocaleString() : 0}
                            </div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#ecfdf5', '#059669')}><Database size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Total Colonnes</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#059669', marginTop: '2px', whiteSpace: 'nowrap' }}>
                                {data.total_columns || 0}
                            </div>
                        </div>
                    </div>

                    <div style={styles.kpiItem}>
                        <div style={styles.iconBg('#fffbeb', '#d97706')}><CheckCircle2 size={18} /></div>
                        <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Version Actuelle</div>
                            <div style={{ fontSize: '14px', fontWeight: '800', color: '#d97706', marginTop: '2px', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>
                                {data.selected_version || selectedVersion}
                            </div>
                        </div>
                    </div>
                </div>

                {/* 3. DATASET DESCRIPTION BANNER */}
                {data.dataset_description && (
                    <div style={{ ...styles.card, padding: '16px 20px', backgroundColor: '#f0f9ff', borderColor: '#bae6fd', display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                        <Info size={18} color="#0284c7" style={{ marginTop: '2px', flexShrink: 0 }} />
                        <div>
                            <h4 style={{ margin: 0, fontSize: '13px', fontWeight: 'bold', color: '#0369a1' }}>Description du jeu de données</h4>
                            <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: '#0284c7', lineHeight: '1.4' }}>{data.dataset_description}</p>
                        </div>
                    </div>
                )}

                {/* 4. METADATA & SCHEMA TABLE */}
                <div style={styles.card}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                        <Sliders size={18} color="#2563eb" />
                        <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#0f172a' }}>
                            Détails du Schéma & Métadonnées des Colonnes
                        </h2>
                    </div>

                    <div style={{ overflowX: 'auto' }}>
                        <table style={styles.table}>
                            <thead>
                                <tr>
                                    <th style={styles.th}>Nom de Colonne</th>
                                    <th style={styles.th}>Type UI</th>
                                    <th style={styles.th}>Type BDD Origine</th>
                                    <th style={styles.th}>Description Détaillée</th>
                                    <th style={styles.th}>Propriétaire</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.columns && data.columns.map((col, idx) => {
                                    const meta = getColumnMeta(col.column_name);

                                    return (
                                        <tr key={idx}>
                                            <td style={{ ...styles.td, fontFamily: 'monospace', fontWeight: 'bold', color: '#0f172a' }}>
                                                {col.column_name}
                                            </td>
                                            <td style={styles.td}>
                                                {renderTypeBadge(col.column_type)}
                                            </td>
                                            <td style={{ ...styles.td, fontFamily: 'monospace', color: '#64748b', fontSize: '12px' }}>
                                                {col.raw_data_type || 'N/A'}
                                            </td>
                                            <td style={{ ...styles.td, color: '#334155', maxWidth: '380px', lineHeight: '1.4' }}>
                                                {meta.description}
                                            </td>
                                            <td style={{ ...styles.td, color: '#64748b', fontSize: '12px' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                    <UserCheck size={14} color="#94a3b8" />
                                                    <span>{meta.owner}</span>
                                                </div>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>
        </div>
    );
}