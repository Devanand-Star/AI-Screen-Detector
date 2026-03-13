import { useState, useRef } from 'react'
import './App.css'

function SeverityIcon({ severity }) {
    const icons = { None: '✅', Minor: '⚠️', Moderate: '🔶', Severe: '🚨' }
    return <span>{icons[severity] || '🔍'}</span>
}

function DamageGauge({ pct, color }) {
    const radius = 54
    const circumference = 2 * Math.PI * radius
    const offset = circumference - (pct / 100) * circumference
    const colors = {
        green: '#22c55e',
        orange: '#f97316',
        amber: '#f59e0b',
        red: '#f43f5e',
    }
    const strokeColor = colors[color] || '#38bdf8'
    return (
        <div className="gauge-wrapper">
            <svg viewBox="0 0 120 120" className="gauge-svg">
                <circle cx="60" cy="60" r={radius} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="10" />
                <circle
                    cx="60" cy="60" r={radius}
                    fill="none"
                    stroke={strokeColor}
                    strokeWidth="10"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    transform="rotate(-90 60 60)"
                    style={{ transition: 'stroke-dashoffset 1.5s cubic-bezier(0.16,1,0.3,1)', filter: `drop-shadow(0 0 6px ${strokeColor})` }}
                />
                <text x="60" y="56" textAnchor="middle" fill="#f8fafc" fontSize="18" fontWeight="800">{pct}%</text>
                <text x="60" y="74" textAnchor="middle" fill="#94a3b8" fontSize="9">Damage</text>
            </svg>
        </div>
    )
}

function InfoRow({ label, value, highlight }) {
    return (
        <div className="info-row">
            <span className="info-label">{label}</span>
            <span className={`info-value ${highlight ? 'info-value-highlight' : ''}`}>{value}</span>
        </div>
    )
}

function App() {
    const [file, setFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const fileInputRef = useRef(null)

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0]
        if (selectedFile && selectedFile.type.startsWith('image/')) {
            processFile(selectedFile)
        } else {
            setError('Please select a valid image file.')
        }
    }

    const processFile = (file) => {
        setFile(file)
        setError(null)
        setResult(null)
        const reader = new FileReader()
        reader.onloadend = () => setPreview(reader.result)
        reader.readAsDataURL(file)
    }

    const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation() }

    const handleDrop = (e) => {
        e.preventDefault(); e.stopPropagation()
        const droppedFile = e.dataTransfer.files[0]
        if (droppedFile && droppedFile.type.startsWith('image/')) processFile(droppedFile)
    }

    const analyzeScreen = async () => {
        if (!file) return
        setLoading(true)
        setError(null)
        const formData = new FormData()
        formData.append('file', file)
        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData,
            })
            if (!response.ok) throw new Error('Backend server is unreachable or returned an error.')
            const data = await response.json()
            setResult(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null) }

    const urgencyClass = (u) => ({ High: 'urgency-high', Medium: 'urgency-medium', Low: 'urgency-low', None: 'urgency-none' }[u] || '')

    return (
        <div className="container">
            <header>
                <div className="logo">🛡️</div>
                <h1>ScreenGuard AI</h1>
                <p className="subtitle">AI-Powered Screen Damage Detection & Analysis</p>
            </header>

            <main className="upload-card">
                {!preview ? (
                    <div
                        className="drop-zone"
                        onClick={() => fileInputRef.current.click()}
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                    >
                        <div className="drop-zone-icon">📁</div>
                        <p>Drag &amp; drop or <strong>click to browse</strong></p>
                        <p style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.5rem' }}>JPEG, PNG or WEBP</p>
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} hidden accept="image/*" />
                    </div>
                ) : (
                    <div className="preview-container">
                        <img src={preview} alt="Upload Preview" className="preview-img" />

                        {!result && !loading && (
                            <button className="btn btn-primary" onClick={analyzeScreen}>
                                🔍 Analyze Screen
                            </button>
                        )}

                        {loading && (
                            <button className="btn btn-primary" disabled>
                                <span className="loading-spinner"></span>
                                Analyzing...
                            </button>
                        )}

                        {result && (
                            <div className="result-card">
                                {/* TOP: Status Badge */}
                                <div className="result-top">
                                    <div className={`badge ${result.prediction === 'Cracked' ? 'badge-red' : 'badge-green'}`}>
                                        <SeverityIcon severity={result.severity} />
                                        &nbsp;{result.prediction === 'Cracked' ? 'Screen Damaged' : 'Screen Healthy'}
                                    </div>
                                    <div className={`urgency-tag ${urgencyClass(result.urgency)}`}>
                                        Urgency: {result.urgency}
                                    </div>
                                </div>

                                {/* GAUGE + KEY STATS */}
                                <div className="stats-row">
                                    <DamageGauge pct={result.damage_percentage} color={result.severity_color} />
                                    <div className="stats-grid">
                                        <InfoRow label="Severity" value={result.severity} highlight />
                                        <InfoRow label="AI Confidence" value={result.confidence} />
                                        <InfoRow label="Recommendation" value={result.recommendation} highlight />
                                        <InfoRow label="Est. Repair Cost" value={result.estimated_repair_cost} />
                                        <InfoRow label="Screen Lifespan" value={result.estimated_screen_life} />
                                    </div>
                                </div>

                                {/* AI Confidence bar */}
                                <div className="confidence-container">
                                    <div className="confidence-label">
                                        <span>AI Confidence Score</span>
                                        <span>{result.confidence}</span>
                                    </div>
                                    <div className="progress-bar-bg">
                                        <div className="progress-bar-fill" style={{ width: result.confidence }}></div>
                                    </div>
                                </div>

                                {/* Action Message */}
                                <div className="action-box">
                                    <div className="action-title">📋 Analysis Summary</div>
                                    <p className="action-text">{result.action_message}</p>
                                </div>

                                {/* Care Tips */}
                                <div className="tips-box">
                                    <div className="tips-title">💡 Care &amp; Action Tips</div>
                                    <ul className="tips-list">
                                        {result.care_tips.map((tip, i) => (
                                            <li key={i} className="tips-item">
                                                <span className="tip-dot"></span>
                                                {tip}
                                            </li>
                                        ))}
                                    </ul>
                                </div>

                                <button className="btn btn-outline" onClick={reset}>
                                    🔄 Analyze Another Image
                                </button>
                            </div>
                        )}
                    </div>
                )}

                {error && <div className="error-msg">⚠️ {error}</div>}
            </main>

            <footer>
                &copy; 2026 ScreenGuard AI | Built with React &amp; TensorFlow
            </footer>
        </div>
    )
}

export default App
