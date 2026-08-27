import React, { useState } from 'react';
import FileUploadModule from './components/FileUploadModule';
import TimeframeGenerator from './components/TimeframeGenerator';

export default function DataUploadPage() {
    const [selectedTable, setSelectedTable] = useState("fact_attraction_hourly");

    return (
        <div style={{
            maxWidth: '1200px',
            width: '100%',
            margin: '20px auto',
            padding: '20px',
            fontFamily: 'sans-serif',
            boxSizing: 'border-box',
            overflowX: 'hidden' // Chặn tuyệt đối cuộn trang ngoài ý muốn
        }}>
            <h1 style={{ color: '#0f172a', marginBottom: '30px' }}>Gestion du Pipeline de Données</h1>

            {/* Module 1: Ingestion */}
            <FileUploadModule onUploadSuccess={(tableName) => setSelectedTable(tableName)} />

            {/* Module 2: Fact Shells */}
            <TimeframeGenerator onTimeframeGenerated={(tableName) => setSelectedTable(tableName)} />
        </div>
    );
}