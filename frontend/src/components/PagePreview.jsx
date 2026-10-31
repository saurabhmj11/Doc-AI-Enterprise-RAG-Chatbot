import { useEffect, useState } from 'react'
import api from '../services/api'

/** Modal showing the cited PDF page(s) with the chunk's bounding box highlighted. */
export default function PagePreview({ docId, pages, bboxes, title, onClose }) {
    const [i, setI] = useState(0)
    const page = pages[i]

    useEffect(() => {
        const onKey = (e) => { if (e.key === 'Escape') onClose() }
        document.addEventListener('keydown', onKey)
        return () => document.removeEventListener('keydown', onKey)
    }, [onClose])

    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div className="rag-preview" onClick={(e) => e.stopPropagation()}>
                <div className="rag-preview-head">
                    <strong>{title} · page {page}{pages.length > 1 ? ` (${i + 1}/${pages.length})` : ''}</strong>
                    <span>
                        {pages.length > 1 && (
                            <>
                                <button onClick={() => setI((i - 1 + pages.length) % pages.length)}>‹</button>
                                <button onClick={() => setI((i + 1) % pages.length)}>›</button>
                            </>
                        )}
                        <button onClick={onClose}>✕</button>
                    </span>
                </div>
                <img src={api.pageUrl(docId, page, bboxes?.[String(page)])} alt={`Page ${page}`} />
            </div>
        </div>
    )
}
