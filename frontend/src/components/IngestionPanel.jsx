import { useRef, useState } from 'react'
import api from '../services/api'

const STAGE_LABEL = { extract: 'Extract + OCR', chunk: 'Chunk', embed: 'Embed', index: 'Vector DB' }

/** Left-hand card: upload PDFs into the large-corpus index with live per-stage progress. */
export function IngestUploader({ status, onIngested }) {
    const inputRef = useRef(null)
    const [jobs, setJobs] = useState([])

    const upload = async (file) => {
        const local = { key: `${file.name}-${Date.now()}`, filename: file.name, state: 'uploading', progress: {} }
        setJobs((j) => [local, ...j])
        const patch = (p) => setJobs((j) => j.map((x) => (x.key === local.key ? { ...x, ...p } : x)))
        try {
            const fd = new FormData()
            fd.append('file', file)
            const { job_id } = await api.request('/rag/ingest', { method: 'POST', body: fd })
            const poll = async () => {
                const job = await api.request(`/rag/jobs/${job_id}`)
                patch(job)
                if (job.state === 'done' || job.state === 'error') onIngested?.()
                else setTimeout(poll, 700)
            }
            poll()
        } catch (err) {
            patch({ state: 'error', error: err.message })
        }
    }

    const onFiles = (files) => Array.from(files).filter((f) => f.name.toLowerCase().endsWith('.pdf')).forEach(upload)
    const docs = status?.documents || []

    return (
        <div className="card">
            <div className="card-header">
                <div className="card-icon upload">📤</div>
                <h2>Ingest & Preprocess</h2>
            </div>
            <div className="upload-area" onClick={() => inputRef.current?.click()}
                onDrop={(e) => { e.preventDefault(); onFiles(e.dataTransfer.files) }} onDragOver={(e) => e.preventDefault()}>
                <input ref={inputRef} type="file" multiple accept="application/pdf" style={{ display: 'none' }}
                    onChange={(e) => { onFiles(e.target.files); e.target.value = '' }} />
                <div className="upload-icon">📁</div>
                <p className="upload-text"><strong>Click to add PDFs</strong> or drag & drop<br />Scanned pages and large images are OCR'd automatically</p>
            </div>

            {jobs.map((j) => (
                <div key={j.key} className="rag-job">
                    <div className="rag-job-head"><span>{j.filename}</span><span className={j.state === 'error' ? 'rag-error' : 'rag-muted'}>{j.state}{j.error ? `: ${j.error}` : ''}</span></div>
                    {(j.stages || []).map((s) => {
                        const [d, n] = j.progress[s] || [0, 0]
                        return (
                            <div key={s} className="rag-job-stage">
                                <span>{STAGE_LABEL[s]}</span>
                                <div className="rag-bar"><span style={{ width: `${n ? (d / n) * 100 : 0}%`, background: 'var(--primary)' }} /></div>
                                <span>{n ? `${d}/${n}` : ''}</span>
                            </div>
                        )
                    })}
                    {j.result && <div className="rag-muted">{j.result.pages} pages ({j.result.ocr_pages} OCR) → {j.result.chunks} chunks</div>}
                </div>
            ))}

            <div className="rag-doclist">
                {docs.map((d) => (
                    <div key={d.doc_id} className="file-item rag-docitem">
                        <span>📑</span>
                        <span className="rag-docname" title={d.filename}>{d.filename}</span>
                        <span className="rag-muted">{d.pages}p · {d.chunks} ch{d.ocr_pages ? ` · ${d.ocr_pages} OCR` : ''}</span>
                    </div>
                ))}
                {!docs.length && <p className="rag-muted">No PDFs indexed yet.</p>}
            </div>
        </div>
    )
}

/** Tab view: the ingestion pipeline with real counters, plus per-document statistics. */
export function IngestionPipeline({ status }) {
    if (!status) return <p className="rag-muted">Loading…</p>
    const t = status.totals
    const docs = status.documents
    const avgTok = t.chunks ? Math.round(docs.reduce((a, d) => a + d.avg_chunk_tokens * d.chunks, 0) / t.chunks) : 0
    const headers = docs.reduce((a, d) => a + (d.headers_removed || 0), 0)
    const embedS = docs.reduce((a, d) => a + d.timings_s.embed, 0)
    const langs = [...new Set(docs.map((d) => d.lang))].join(', ') || '–'
    const steps = [
        ['📄', 'PDFs', t.documents, `${t.size_mb} MB`],
        ['🔍', 'Extract + OCR', t.pages.toLocaleString(), `pages · ${t.ocr_pages} OCR'd · ${t.ocr_images} images OCR'd`],
        ['🧹', 'Clean', headers.toLocaleString(), `header/footer blocks removed · lang ${langs}`],
        ['✂️', 'Chunk', t.chunks.toLocaleString(), `~${avgTok} tok avg · max ${status.chunking.tokens} / overlap ${status.chunking.overlap}`],
        ['🧮', 'Embed', status.embed_model.split('/').pop(), `${docs[0]?.embed_dim || '–'}-d · ${Math.round(embedS)}s CPU`],
        ['🗄️', 'Vector DB', t.vectors.toLocaleString(), status.vector_db],
    ]
    return (
        <div className="eval-dashboard">
            <div className="rag-pipeline">
                {steps.map(([icon, k, v, d], i) => (
                    <div key={k} className="rag-step-wrap">
                        {i > 0 && <div className="arch-arrow">→</div>}
                        <div className="eval-card rag-step">
                            <div className="eval-card-title">{icon} {k}</div>
                            <div className="eval-card-value">{v}</div>
                            <div className="eval-card-sub">{d}</div>
                        </div>
                    </div>
                ))}
            </div>
            <div className="rag-table-wrap">
                <table className="eval-table">
                    <thead>
                        <tr><th>File</th><th>Pages</th><th>Native</th><th>OCR pages</th><th>Images OCR'd</th><th>Chunks</th><th>Lang</th><th>Extract</th><th>Embed</th><th>Index</th></tr>
                    </thead>
                    <tbody>
                        {docs.map((d) => (
                            <tr key={d.doc_id}>
                                <td>{d.filename}</td><td>{d.pages}</td><td>{d.native_pages}</td><td>{d.ocr_pages}</td><td>{d.ocr_images}</td>
                                <td>{d.chunks}</td><td>{d.lang}</td><td>{d.timings_s.extract}s</td><td>{d.timings_s.embed}s</td><td>{d.timings_s.index}s</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
