import { useState, useRef, useEffect, useCallback } from 'react'
import SettingsModal from './components/SettingsModal'
import Login from './components/Login'
import RagChat, { fmtMs } from './components/RagChat'
import EvalPanel from './components/EvalPanel'
import { IngestUploader, IngestionPipeline } from './components/IngestionPanel'
import api from './services/api'

function App() {
    // State management
    const [user, setUser] = useState(null)
    const [authChecked, setAuthChecked] = useState(false)
    const [activeTab, setActiveTab] = useState('chat') // 'chat' | 'ingest' | 'eval' | 'extract'
    const [showSettings, setShowSettings] = useState(false)

    // Large-corpus RAG state
    const [ragStatus, setRagStatus] = useState(null)
    const [liveMetrics, setLiveMetrics] = useState(null)
    const [evalRefresh, setEvalRefresh] = useState(0)

    // Legacy structured extraction (logistics docs) state
    const [files, setFiles] = useState([])
    const [processedDocs, setProcessedDocs] = useState([])
    const [uploading, setUploading] = useState(false)
    const [extraction, setExtraction] = useState(null)
    const [extracting, setExtracting] = useState(false)

    const fileInputRef = useRef(null)

    const refreshRag = useCallback(async () => {
        try {
            const [status, metrics] = await Promise.all([api.request('/rag/status'), api.request('/rag/metrics')])
            setRagStatus(status)
            setLiveMetrics(metrics)
            return status
        } catch (err) {
            console.error('RAG status error:', err)
        }
    }, [])

    // Load auth, corpus status and legacy documents
    useEffect(() => {
        const init = async () => {
            if (api.getToken()) {
                try {
                    const userData = await api.getCurrentUser()
                    setUser(userData)
                    await refreshRag()

                    const corpusData = await api.request('/documents')
                    if (corpusData?.documents) {
                        setProcessedDocs(corpusData.documents.map(d => ({
                            document_id: d.document_id,
                            filename: d.filename,
                            chunks_created: d.chunk_count,
                            file_type: d.file_type
                        })))
                    }
                } catch (err) {
                    console.error('Initialization error:', err)
                    if (err.message === 'Unauthorized') {
                        api.logout()
                    }
                }
            }
            setAuthChecked(true)
        }
        init()
    }, [refreshRag])

    // Poll while the models are warming up
    useEffect(() => {
        if (!user || ragStatus?.ready) return
        const t = setTimeout(refreshRag, 3000)
        return () => clearTimeout(t)
    }, [user, ragStatus, refreshRag])

    const onAnswered = () => {
        refreshRag()
        setEvalRefresh(k => k + 1)
    }

    // ---- legacy extraction ---------------------------------------------------
    const handleUpload = async () => {
        if (files.length === 0) return
        setUploading(true)
        try {
            const formData = new FormData()
            files.forEach(f => formData.append('files', f))
            const result = await api.request('/upload/batch', { method: 'POST', body: formData })
            if (result?.results) {
                setProcessedDocs(prev => [...prev, ...result.results])
            }
            setFiles([])
        } catch (error) {
            setExtraction({ error: `Upload failed: ${error.message}` })
        } finally {
            setUploading(false)
        }
    }

    const handleDeleteDocument = async (documentId) => {
        try {
            await api.request(`/documents/${documentId}`, { method: 'DELETE' })
            setProcessedDocs(prev => prev.filter(d => d.document_id !== documentId))
            if (extraction && extraction.document_id === documentId) {
                setExtraction(null)
            }
        } catch (error) {
            console.error('Error deleting document:', error)
        }
    }

    const handleExtract = async () => {
        if (processedDocs.length === 0 || extracting) return
        const targetDoc = processedDocs[processedDocs.length - 1]
        setExtracting(true)
        try {
            const data = await api.request('/extract', {
                method: 'POST',
                body: JSON.stringify({ document_id: targetDoc.document_id })
            })
            setExtraction(data)
        } catch (error) {
            setExtraction({ error: error.message })
        } finally {
            setExtracting(false)
        }
    }

    const handleLogout = () => {
        api.logout()
        setUser(null)
        window.location.reload()
    }

    if (!authChecked) return null
    if (!user) return <Login onLoginSuccess={setUser} />

    const t = ragStatus?.totals
    const p95 = liveMetrics?.latency_ms?.total_ms?.p95

    return (
        <div className="app">
            {showSettings && (
                <SettingsModal onClose={() => setShowSettings(false)} />
            )}

            <header className="header" style={{ position: 'relative' }}>
                <button
                    className="btn-settings"
                    onClick={() => setShowSettings(true)}
                    style={{
                        position: 'absolute',
                        top: 0,
                        right: 0,
                        background: 'transparent',
                        border: '1px solid var(--border)',
                        color: 'var(--text-secondary)',
                        padding: '8px 16px',
                        borderRadius: 'var(--radius-sm)',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        fontSize: '0.9rem',
                        transition: 'all 0.2s'
                    }}
                >
                    ⚙️ Settings
                </button>
                <button
                    className="btn-logout"
                    onClick={handleLogout}
                    style={{
                        position: 'absolute',
                        top: 0,
                        right: '110px',
                        background: 'transparent',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        color: '#fca5a5',
                        padding: '8px 16px',
                        borderRadius: 'var(--radius-sm)',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        fontSize: '0.9rem',
                        transition: 'all 0.2s'
                    }}
                >
                    🔒 Logout
                </button>
                <h1>⚡ <span className="gradient">Doc-AI Enterprise RAG Chatbot</span></h1>
                <p>Question answering over a large PDF corpus, using open-source embeddings, HNSW indexing and page-level citations.</p>
            </header>

            {/* Corpus Banner (live values from /api/rag/status) */}
            <div className="corpus-banner">
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                    <span className="corpus-pill">📚 {t ? t.documents : '–'} PDFs</span>
                    <span className="corpus-pill">📄 {t ? t.pages.toLocaleString() : '–'} pages</span>
                    <span className="corpus-pill">🧩 {t ? t.vectors.toLocaleString() : '–'} vectors</span>
                    {ragStatus && <span className="corpus-pill">🧮 {ragStatus.embed_model.split('/').pop()}</span>}
                    <span className="corpus-pill">🗄️ ChromaDB HNSW + BM25 (RRF)</span>
                    {ragStatus?.rerank_model && <span className="corpus-pill">⚖️ cross-encoder rerank</span>}
                    {ragStatus && <span className="corpus-pill">🤖 {ragStatus.provider}:{ragStatus.llm_model}</span>}
                    {ragStatus && !ragStatus.ready && <span className="corpus-pill">⏳ warming up…</span>}
                </div>
                {p95 != null && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Target: 2–5s</span>
                        <span className={`latency-pill ${p95 <= 5000 ? 'highlight' : 'warn'}`}>p95: {fmtMs(p95)}</span>
                    </div>
                )}
            </div>

            <div className="main-layout">
                {/* Left Panel - PDF ingestion into the large-corpus index */}
                <div className="left-panel">
                    <IngestUploader status={ragStatus} onIngested={refreshRag} />
                </div>

                {/* Right Panel - Chat, Ingestion pipeline, Benchmark, Extraction */}
                <div className="right-panel">
                    <div className="card">
                        <div className="tabs">
                            <button className={`tab ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
                                💬 Corpus Q&A
                            </button>
                            <button className={`tab ${activeTab === 'ingest' ? 'active' : ''}`} onClick={() => { setActiveTab('ingest'); refreshRag() }}>
                                🗂️ Ingestion Pipeline
                            </button>
                            <button className={`tab ${activeTab === 'eval' ? 'active' : ''}`} onClick={() => { setActiveTab('eval'); setEvalRefresh(k => k + 1) }}>
                                📈 Benchmarking & Metrics
                            </button>
                            <button className={`tab ${activeTab === 'extract' ? 'active' : ''}`} onClick={() => setActiveTab('extract')}>
                                📊 Extract Data
                            </button>
                        </div>

                        {/* Kept mounted so the conversation survives tab switches */}
                        <div style={{ display: activeTab === 'chat' ? 'block' : 'none' }}>
                            <RagChat status={ragStatus} onAnswered={onAnswered} />
                        </div>

                        {activeTab === 'ingest' && <IngestionPipeline status={ragStatus} />}

                        {activeTab === 'eval' && <EvalPanel refreshKey={evalRefresh} />}

                        {/* Structured extraction for logistics documents (legacy pipeline) */}
                        {activeTab === 'extract' && (
                            <div className="extraction-panel">
                                <p>Extract structured shipment fields (shipper, consignee, dates, rate…) from a logistics document into JSON.</p>
                                <div
                                    className="upload-area"
                                    style={{ margin: '12px 0', padding: '18px' }}
                                    onClick={() => fileInputRef.current?.click()}
                                    onDrop={(e) => { e.preventDefault(); setFiles(prev => [...prev, ...Array.from(e.dataTransfer.files)]) }}
                                    onDragOver={(e) => e.preventDefault()}
                                >
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        multiple
                                        accept=".pdf,.docx,.txt"
                                        onChange={(e) => setFiles(prev => [...prev, ...Array.from(e.target.files)])}
                                        style={{ display: 'none' }}
                                    />
                                    <p className="upload-text"><strong>Add documents</strong> (PDF, DOCX, TXT)</p>
                                </div>

                                {(files.length > 0 || processedDocs.length > 0) && (
                                    <div className="file-list" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '220px', overflowY: 'auto' }}>
                                        {processedDocs.map(doc => (
                                            <div key={doc.document_id} className="file-item rag-docitem">
                                                <span>📑</span>
                                                <span className="rag-docname">{doc.filename}</span>
                                                <span className="rag-muted">{doc.chunks_created} chunks</span>
                                                <button className="rag-x" onClick={() => handleDeleteDocument(doc.document_id)} title="Remove document">×</button>
                                            </div>
                                        ))}
                                        {files.map((f, i) => (
                                            <div key={i} className="file-item rag-docitem">
                                                <span>📄</span>
                                                <span className="rag-docname">{f.name}</span>
                                                <span className="rag-muted">{(f.size / 1024).toFixed(0)} KB</span>
                                                <button className="rag-x" onClick={() => setFiles(prev => prev.filter((_, j) => j !== i))} title="Remove file">×</button>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                <div style={{ display: 'flex', gap: '10px', marginTop: '12px' }}>
                                    <button className="btn btn-primary" onClick={handleUpload} disabled={files.length === 0 || uploading} style={{ width: 'auto', marginTop: 0 }}>
                                        {uploading ? 'Uploading…' : `📤 Upload ${files.length || ''}`}
                                    </button>
                                    <button className="btn btn-primary" onClick={handleExtract} disabled={processedDocs.length === 0 || extracting} style={{ width: 'auto', marginTop: 0 }}>
                                        {extracting ? 'Extracting...' : '🔍 Run Extraction'}
                                    </button>
                                </div>

                                {extraction?.error && <p className="rag-error" style={{ marginTop: '12px' }}>⚠️ {extraction.error}</p>}
                                {extraction && !extraction.error && (
                                    <div className="extraction-grid" style={{ marginTop: '20px' }}>
                                        {Object.entries(extraction.extraction).map(([key, value]) => (
                                            <div key={key} className="extraction-field">
                                                <label>{key.replace(/_/g, ' ')}</label>
                                                <div className={`value ${value === null ? 'missing' : ''}`}>
                                                    {value !== null ? String(value) : 'Not found'}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default App
