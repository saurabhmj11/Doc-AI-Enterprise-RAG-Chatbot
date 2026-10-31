import { useEffect, useRef, useState } from 'react'
import api from '../services/api'
import PagePreview from './PagePreview'

const EXAMPLES = [
    'What is the difference between TD learning and Monte Carlo methods?',
    'Which NIST control family covers incident response?',
    'How does Python handle mutable default arguments?',
    'What is a technical review in the NASA systems engineering process?',
]

export const fmtMs = (ms) => (ms == null ? '–' : ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`)
export const pct = (x) => (x == null ? '–' : `${(x * 100).toFixed(1)}%`)
const pageLabel = (c) => (c.page_start === c.page_end ? `p. ${c.page_start}` : `pp. ${c.page_start}–${c.page_end}`)

/** Large-corpus chat: streamed answers with [S#] citations + a retrieval trace for the selected turn. */
export default function RagChat({ status, onAnswered }) {
    const [turns, setTurns] = useState([])
    const [selected, setSelected] = useState(null)
    const [question, setQuestion] = useState('')
    const [busy, setBusy] = useState(false)
    const [preview, setPreview] = useState(null)
    const endRef = useRef(null)

    useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [turns])

    const ask = async (q) => {
        q = q.trim()
        if (!q || busy) return
        setQuestion('')
        setBusy(true)
        const idx = turns.length
        const history = turns.filter((t) => t.done).slice(-2)
            .flatMap((t) => [{ role: 'user', content: t.question }, { role: 'assistant', content: t.answer }])
        setTurns((prev) => [...prev, { question: q, answer: '', retrieval: null, done: null, error: null }])
        setSelected(idx)
        const update = (patch) => setTurns((prev) => prev.map((t, i) => (i === idx ? { ...t, ...(typeof patch === 'function' ? patch(t) : patch) } : t)))
        try {
            await api.stream('/rag/chat', { question: q, history }, (ev) => {
                if (ev.type === 'retrieval') update({ retrieval: ev })
                else if (ev.type === 'token') update((t) => ({ answer: t.answer + ev.text }))
                else if (ev.type === 'error') update({ error: ev.message })
                else if (ev.type === 'notice') update({ notice: ev.message })
                else if (ev.type === 'done') update({ done: ev, answer: ev.answer })
            })
        } catch (err) {
            update({ error: err.message })
        } finally {
            setBusy(false)
            onAnswered?.()
        }
    }

    const openSource = (s) => setPreview({ docId: s.doc_id, pages: s.pages, bboxes: s.bboxes, title: `${s.tag || ''} ${s.filename}`.trim() })
    const turn = selected != null ? turns[selected] : null

    return (
        <div className="rag-chat">
            <div className="chat-container rag-chat-main">
                <div className="chat-messages">
                    {turns.length === 0 ? (
                        <div className="empty-state">
                            <div className="icon">⚡</div>
                            <h3>Ask the PDF library</h3>
                            <p>Answers come only from the {status?.totals?.documents ?? ''} indexed PDFs, citing file and page. Click a citation to open the page.</p>
                            <div className="rag-examples">
                                {EXAMPLES.map((q) => (
                                    <button key={q} className="rag-example" onClick={() => ask(q)} disabled={busy}>{q}</button>
                                ))}
                            </div>
                        </div>
                    ) : turns.map((t, i) => (
                        <div key={i}>
                            <div className="message message-user"><div className="message-bubble">{t.question}</div></div>
                            <div className={`message message-assistant ${selected === i ? 'rag-selected' : ''}`} onClick={() => setSelected(i)}>
                                <div className="message-bubble">
                                    <Answer turn={t} onCite={openSource} />
                                </div>
                            </div>
                        </div>
                    ))}
                    <div ref={endRef} />
                </div>
                <div className="chat-input-container">
                    <input
                        type="text"
                        className="chat-input"
                        placeholder={status?.totals?.vectors ? 'Ask any question across the PDF corpus…' : 'Index PDFs first…'}
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter') ask(question) }}
                        disabled={busy || !status?.totals?.vectors}
                    />
                    <button className="btn-send" onClick={() => ask(question)} disabled={busy || !question.trim()}>
                        {busy ? <span className="spinner"></span> : '➤'}
                    </button>
                </div>
            </div>
            <RetrievalTrace turn={turn} onOpen={openSource} />
            {preview && <PagePreview {...preview} onClose={() => setPreview(null)} />}
        </div>
    )
}

function Answer({ turn, onCite }) {
    const sources = turn.retrieval?.sources || []
    if (turn.error && !turn.answer) return <div className="message-content rag-error">⚠️ {turn.error}</div>
    if (!turn.retrieval) return <div className="message-content rag-muted">Retrieving…</div>
    if (!turn.answer) return <div className="message-content rag-muted">Generating with {turn.retrieval.model}…</div>

    // render text with clickable [S#] chips
    const parts = turn.answer.split(/(\[S\d+\])/g).map((part, i) => {
        const m = part.match(/^\[S(\d+)\]$/)
        if (!m) return <span key={i}>{part.replace(/\*\*/g, '')}</span>
        const s = sources[+m[1] - 1]
        return s ? (
            <button key={i} className="citation-chip rag-cite" title={`${s.filename} ${s.page_label}`}
                onClick={(e) => { e.stopPropagation(); onCite(s) }}>S{m[1]}</button>
        ) : <span key={i} className="citation-chip rag-cite-bad" title="invalid citation">S{m[1]}?</span>
    })
    const d = turn.done
    const v = d?.verification
    return (
        <>
            <div className="message-content" style={{ whiteSpace: 'pre-line' }}>
                {parts}{!d && <span className="rag-cursor" />}
            </div>
            {d && d.citations.length > 0 && (
                <div className="rag-sources">
                    {d.citations.map((c) => (
                        <button key={c.tag} className="rag-source" onClick={(e) => { e.stopPropagation(); onCite(c) }}>
                            <b>{c.tag}</b> 📄 {c.filename} · {c.page_label}
                        </button>
                    ))}
                </div>
            )}
            {d && (
                <div className="latency-container">
                    <span className={`latency-pill ${d.timings.total_ms <= 5000 ? 'highlight' : 'warn'}`}>⚡ {fmtMs(d.timings.total_ms)} total</span>
                    <span className="latency-pill">first token {fmtMs(d.timings.ttft_ms)}</span>
                    <span className="latency-pill">🔍 retrieval {fmtMs(d.timings.retrieval_ms)}</span>
                    {v.refused ? <span className="latency-pill">declined: not in corpus</span> : (
                        <>
                            <span className={`latency-pill ${v.hallucination_flag ? 'warn' : 'highlight'}`}>grounding {pct(v.grounding_score)}</span>
                            <span className={`latency-pill ${v.citation_validity === 1 ? 'highlight' : 'warn'}`}>{v.citations_valid}/{v.citations_total} citations valid</span>
                        </>
                    )}
                </div>
            )}
            {turn.notice && <div className="rag-muted">ℹ️ {turn.notice}</div>}
            {turn.error && <div className="rag-error">⚠️ {turn.error}</div>}
        </>
    )
}

function Bar({ label, value, frac, color }) {
    return (
        <>
            <span>{label}</span>
            <div className="rag-bar"><span style={{ width: `${Math.max(0, Math.min(1, frac || 0)) * 100}%`, background: color }} /></div>
            <span>{value}</span>
        </>
    )
}

function RetrievalTrace({ turn, onOpen }) {
    const [open, setOpen] = useState({})
    if (!turn?.retrieval) {
        return (
            <aside className="rag-trace">
                <h3>🔎 Retrieval trace</h3>
                <p className="rag-muted">Ask a question to see the latency breakdown and the top retrieved chunks: dense HNSW and BM25 results, fused with RRF, then reranked by a cross-encoder.</p>
            </aside>
        )
    }
    const r = turn.retrieval
    const tm = { ...r.timings, ...(turn.done?.timings || {}) }
    const segs = [
        ['embed', tm.embed_ms, '#64748b'], ['dense HNSW', tm.dense_ms, '#6366f1'], ['BM25', tm.bm25_ms, '#a78bfa'],
        ['rerank', tm.rerank_ms, '#10b981'], ['LLM', tm.generation_ms, '#f59e0b'],
    ]
    const total = tm.total_ms || segs.reduce((a, s) => a + (s[1] || 0), 0)
    const rr = r.candidates.map((c) => c.rerank_score ?? 0)
    const maxRr = Math.max(...rr), minRr = Math.min(...rr)
    const maxBm = Math.max(...r.candidates.map((c) => c.bm25_score || 0), 1e-9)
    const tagOf = Object.fromEntries(r.sources.map((s) => [s.chunk_id, s.tag]))

    return (
        <aside className="rag-trace">
            <h3>⏱ Latency {turn.done ? fmtMs(tm.total_ms) : '…'} <span className="rag-muted">· {r.provider}:{r.model}</span></h3>
            <div className="rag-waterfall">
                {segs.map(([n, v, c]) => <span key={n} title={`${n}: ${fmtMs(v)}`} style={{ width: `${((v || 0) / total) * 100}%`, background: c }} />)}
            </div>
            <div className="rag-legend">
                {segs.map(([n, v, c]) => <span key={n}><i style={{ background: c }} />{n} {fmtMs(v)}</span>)}
                {turn.done && <span>first token {fmtMs(tm.ttft_ms)}</span>}
            </div>
            <h3>🧩 Top retrieved chunks</h3>
            <div className="rag-funnel">
                <span>dense {r.counts.dense}</span><span>BM25 {r.counts.bm25}</span><span>RRF {r.counts.fused}</span>
                <span>reranked {r.counts.reranked}</span><span>context {r.sources.length}</span>
            </div>
            {r.candidates.map((c) => {
                const tag = tagOf[c.chunk_id]
                return (
                    <div key={c.chunk_id} className={`rag-cand ${tag ? 'used' : c.selected ? '' : 'dim'}`}>
                        <div className="rag-cand-head">
                            <button onClick={() => onOpen({ ...c, tag: `#${c.final_rank}` })}>#{c.final_rank} {c.filename.replace(/\.pdf$/i, '').replace(/_/g, ' ')} · {pageLabel(c)}</button>
                            <span>
                                {tag && <span className="rag-badge used">{tag} in context</span>}
                                {c.source !== 'native' && <span className="rag-badge">{c.source}</span>}
                            </span>
                        </div>
                        <div className="rag-bars">
                            {c.rerank_score != null && <Bar label="rerank" value={c.rerank_score.toFixed(2)} frac={(c.rerank_score - minRr) / ((maxRr - minRr) || 1)} color="#10b981" />}
                            <Bar label="dense" value={c.dense_score != null ? `${c.dense_score.toFixed(3)} #${c.dense_rank}` : '–'} frac={c.dense_score} color="#6366f1" />
                            <Bar label="BM25" value={c.bm25_score != null ? `${c.bm25_score.toFixed(1)} #${c.bm25_rank}` : '–'} frac={(c.bm25_score || 0) / maxBm} color="#a78bfa" />
                        </div>
                        <div className={`rag-cand-text ${open[c.chunk_id] ? 'open' : ''}`}
                            onClick={() => setOpen((o) => ({ ...o, [c.chunk_id]: !o[c.chunk_id] }))}>{c.text}</div>
                    </div>
                )
            })}
        </aside>
    )
}
