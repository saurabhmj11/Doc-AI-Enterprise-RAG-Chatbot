import { useEffect, useState } from 'react'
import api from '../services/api'
import { fmtMs, pct } from './RagChat'

const ABLATION = { dense: 'Dense only (HNSW)', bm25: 'BM25 only', hybrid_rrf: 'Hybrid (RRF)', hybrid_rerank: 'Hybrid + cross-encoder' }

function Card({ title, value, sub, color }) {
    return (
        <div className="eval-card">
            <div className="eval-card-title">{title}</div>
            <div className="eval-card-value" style={color ? { color } : undefined}>{value}</div>
            <div className="eval-card-sub">{sub}</div>
        </div>
    )
}

/** Offline eval report (scripts/run_eval.py) + live latency/grounding monitoring from real queries. */
export default function EvalPanel({ refreshKey }) {
    const [report, setReport] = useState(null)
    const [live, setLive] = useState(null)

    useEffect(() => {
        api.request('/rag/eval').then(setReport).catch(() => setReport({ available: false }))
        api.request('/rag/metrics').then(setLive).catch(() => setLive({ count: 0 }))
    }, [refreshKey])

    const ok = '#34d399', bad = '#fbbf24'
    return (
        <div className="eval-dashboard">
            <div className="eval-header">
                <div>
                    <h3 style={{ fontSize: '1.2rem', fontWeight: 600 }}>Offline evaluation</h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                        {report?.available
                            ? `${report.n_questions} seeded questions with gold pages · ${report.provider}:${report.model} · ${report.created}`
                            : <>No report yet. Run <code>python scripts/build_eval.py</code> then <code>python scripts/run_eval.py</code> in backend/.</>}
                    </p>
                </div>
            </div>

            {report?.available && (() => {
                const r = report.retrieval, g = report.generation, l = report.latency_ms
                return (
                    <>
                        <div className="eval-grid">
                            <Card title="p95 latency" value={fmtMs(l.total.p95)} sub={`p50 ${fmtMs(l.total.p50)} · ${pct(l.within_5s)} within 5s`} color={l.total.p95 <= 5000 ? ok : bad} />
                            <Card title="p95 first token" value={fmtMs(l.ttft.p95)} sub={`p50 ${fmtMs(l.ttft.p50)}`} />
                            <Card title="p95 retrieval" value={fmtMs(l.retrieval.p95)} sub={`p50 ${fmtMs(l.retrieval.p50)}`} />
                            <Card title="Recall@5" value={pct(r['recall@5'])} sub={`R@1 ${pct(r['recall@1'])} · R@10 ${pct(r['recall@10'])}`} color="#38bdf8" />
                            <Card title="MRR@10" value={r.mrr.toFixed(3)} sub="rank of first gold-page chunk" color="#a78bfa" />
                            <Card title="Citation accuracy" value={pct(g.citation_accuracy)} sub={`cites the gold page · validity ${pct(g.citation_validity)}`} />
                            <Card title="Hallucination rate" value={pct(g.hallucination_rate)} sub="answers with unsupported claims" color={g.hallucination_rate <= 0.1 ? ok : bad} />
                            <Card title="Answered" value={pct(g.answer_rate)} sub={`avg grounding ${pct(g.avg_grounding)}`} />
                        </div>
                        <table className="eval-table">
                            <thead><tr><th>Retriever ablation</th><th>R@1</th><th>R@3</th><th>R@5</th><th>R@10</th><th>MRR</th></tr></thead>
                            <tbody>
                                {Object.entries(report.ablation).map(([k, m]) => (
                                    <tr key={k}><td>{ABLATION[k] || k}</td><td>{pct(m['recall@1'])}</td><td>{pct(m['recall@3'])}</td>
                                        <td>{pct(m['recall@5'])}</td><td>{pct(m['recall@10'])}</td><td>{m.mrr.toFixed(3)}</td></tr>
                                ))}
                            </tbody>
                        </table>
                        <div className="rag-table-wrap">
                            <table className="eval-table">
                                <thead><tr><th>Question</th><th>Gold</th><th>Rank</th><th>Total</th><th>TTFT</th><th>Cited gold</th><th>Grounding</th></tr></thead>
                                <tbody>
                                    {report.items.map((it) => (
                                        <tr key={it.id}>
                                            <td className="rag-wrap">{it.question}</td>
                                            <td>{it.filename.replace(/\.pdf$/i, '')} p.{it.gold_pages.join(',')}</td>
                                            <td>{it.rank ?? '–'}</td><td>{fmtMs(it.total_ms)}</td><td>{fmtMs(it.ttft_ms)}</td>
                                            <td>{it.cited_gold ? '✓' : '✗'}</td><td>{it.refused ? 'declined' : pct(it.grounding_score)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </>
                )
            })()}

            <div className="eval-header">
                <div>
                    <h3 style={{ fontSize: '1.2rem', fontWeight: 600 }}>Live monitoring</h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>From real queries (backend/data/logs/queries.jsonl)</p>
                </div>
            </div>
            {!live?.count ? <p className="rag-muted">No queries yet.</p> : (
                <>
                    <div className="eval-grid">
                        <Card title="Queries" value={live.count} sub="last 200" />
                        <Card title="p95 total" value={fmtMs(live.latency_ms.total_ms.p95)} sub={`p50 ${fmtMs(live.latency_ms.total_ms.p50)}`} color={live.latency_ms.total_ms.p95 <= 5000 ? ok : bad} />
                        <Card title="p95 first token" value={fmtMs(live.latency_ms.ttft_ms.p95)} sub={`p50 ${fmtMs(live.latency_ms.ttft_ms.p50)}`} />
                        <Card title="Within 5s" value={pct(live.within_5s)} sub="end-to-end" />
                        <Card title="Hallucination flags" value={pct(live.hallucination_flag_rate)} sub={`avg grounding ${pct(live.avg_grounding)}`} />
                    </div>
                    <div className="rag-table-wrap">
                        <table className="eval-table">
                            <thead><tr><th>Time</th><th>Question</th><th>Model</th><th>Total</th><th>TTFT</th><th>Retrieval</th><th>Grounding</th></tr></thead>
                            <tbody>
                                {live.recent.map((q, i) => (
                                    <tr key={i}>
                                        <td>{q.ts.slice(11)}</td><td className="rag-wrap">{q.question}</td><td>{q.model}</td>
                                        <td>{fmtMs(q.timings.total_ms)}</td><td>{fmtMs(q.timings.ttft_ms)}</td><td>{fmtMs(q.timings.retrieval_ms)}</td>
                                        <td>{q.refused ? 'declined' : pct(q.grounding_score)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            )}
        </div>
    )
}
