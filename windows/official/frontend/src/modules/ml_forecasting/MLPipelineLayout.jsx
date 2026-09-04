import React from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';

export default function MLPipelineLayout() {
    const location = useLocation();

    // Déclaration des sous-modules du pipeline Machine Learning
    const subModules = [
        { path: 'split', label: '1. Séparation des Données', desc: 'Partitionnement Train / Val / Test' },
        { path: 'training', label: '2. Entraînement des Modèles', desc: 'Entraînement des modèles ML & DL' },
        { path: 'evaluation', label: '3. Évaluation des Performances', desc: 'Comparaison des métriques et erreurs' },
        { path: 'explainability', label: '4. Explicabilité (XAI)', desc: 'Analyse d\'impact SHAP & Feature Importance' },
        { path: 'hyperparameters', label: '5. Optimisation des Hyperparamètres', desc: 'Recherche sur grille (Grid/Random Search)' },
        { path: 'registry', label: '6. Registre des Modèles', desc: 'Gestion des versions et artefacts' },
        { path: 'deployment', label: '7. Déploiement & Prédiction', desc: 'Inférence en temps réel et prévisions' }
    ];

    return (
        <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
            {/* En-tête du Module */}
            <div style={{ marginBottom: '24px' }}>
                <h1 style={{ fontSize: '24px', fontWeight: 'bold', color: '#0f172a' }}>
                    Module Pipeline Machine Learning
                </h1>
                <p style={{ color: '#64748b', fontSize: '14px' }}>
                    Entraînement, évaluation, explicabilité et gestion du cycle de vie des modèles prédictifs.
                </p>
            </div>

            {/* Barre de Navigation / Onglets */}
            <div style={{ 
                display: 'flex', 
                gap: '8px', 
                borderBottom: '2px solid #e2e8f0', 
                marginBottom: '24px',
                overflowX: 'auto',
                whiteSpace: 'nowrap',
                paddingBottom: '4px'
            }}>
                {subModules.map((tab) => (
                    <NavLink
                        key={tab.path}
                        to={tab.path}
                        title={tab.desc}
                        style={({ isActive }) => ({
                            padding: '12px 18px',
                            textDecoration: 'none',
                            fontWeight: isActive ? '600' : '500',
                            fontSize: '14px',
                            color: isActive ? '#4f46e5' : '#64748b',
                            borderBottom: isActive ? '3px solid #4f46e5' : '3px solid transparent',
                            marginBottom: '-6px',
                            transition: 'all 0.2s ease-in-out',
                            borderRadius: '6px 6px 0 0',
                            backgroundColor: isActive ? '#f8fafc' : 'transparent'
                        })}
                    >
                        {tab.label}
                    </NavLink>
                ))}
            </div>

            {/* Zone d'affichage des composants enfants (Routes filles) */}
            <div style={{ backgroundColor: '#ffffff', borderRadius: '8px', minHeight: '500px' }}>
                <Outlet />
            </div>
        </div>
    );
}