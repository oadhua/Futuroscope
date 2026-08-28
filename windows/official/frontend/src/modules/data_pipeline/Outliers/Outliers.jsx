import React, { useState } from 'react';
import OutliersModule from '../Outliers/components/Outliers';

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
            {/* Module 4: Single Column Analysis */}
            <OutliersModule/>
        </div>
    );
}