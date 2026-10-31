"""
Vector Store Module
FINAL Production Version
"""

from typing import Optional, TYPE_CHECKING
import os
import json
import threading
from pathlib import Path
import concurrent.futures

from config import get_settings

if TYPE_CHECKING:
    from core.document_processor import DocumentChunk

settings = get_settings()


class VectorStore:

    def __init__(self):

        import chromadb

        os.makedirs(settings.chroma_persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )

        self.metadata_file = Path(settings.chroma_persist_dir) / "document_metadata.json"
        self._metadata_lock = threading.Lock()

        self._document_metadata = self._load_metadata()

        # Reusable executor
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    # =================================================

    def _load_metadata(self):

        try:
            if self.metadata_file.exists():
                return json.loads(self.metadata_file.read_text())
        except Exception:
            pass

        return {}

    def _save_metadata(self):

        with self._metadata_lock:

            temp_file = self.metadata_file.with_suffix(".tmp")

            temp_file.write_text(json.dumps(self._document_metadata, indent=2))

            temp_file.replace(self.metadata_file)

    # =================================================

    def add_document(self, document_id, chunks, filename, file_type, file_path):

        # Single collection strategy
        collection_name = "documents"

        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for c in chunks:

            if not c.embedding:
                raise ValueError("Chunk missing embedding")

            ids.append(c.chunk_id)
            embeddings.append(c.embedding)
            documents.append(c.text)

            metadatas.append({
                "document_id": c.document_id,
                "page": c.page or 0,
                "chunk_index": c.chunk_index
            })

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        self._document_metadata[document_id] = {
            "filename": filename,
            "file_type": file_type,
            "file_path": file_path,
            "chunk_count": len(chunks),
            "collection_name": "documents"
        }

        self._save_metadata()

    # =================================================

    def search(self, document_id, query_embedding, top_k=10):

        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        if document_id not in self._document_metadata:
            return []

        collection_name = "documents"

        collection = self.client.get_collection(collection_name)

        # Filter by document_id
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 100),
            where={"document_id": document_id},
            include=["documents", "metadatas", "distances"]
        )

        output = []

        for i in range(len(results["ids"][0])):

            distance = results["distances"][0][i]

            # Correct cosine similarity conversion
            # Distance 0 = Identical, 2 = Opposite
            similarity = 1 - (distance / 2)
            similarity = max(0.0, min(1.0, similarity))

            output.append({
                "chunk_id": results["ids"][0][i],
                "document_id": document_id,
                "filename": self._document_metadata[document_id]["filename"],
                "text": results["documents"][0][i],
                "page": results["metadatas"][0][i].get("page"),
                "chunk_index": results["metadatas"][0][i].get("chunk_index"),
                "similarity_score": similarity
            })

        return output

    # =================================================

    def list_documents(self):
        """List all indexed documents with metadata."""
        return [
            {
                "document_id": doc_id,
                "filename": meta.get("filename", "unknown"),
                "file_type": meta.get("file_type", ""),
                "chunk_count": meta.get("chunk_count", 0),
                "file_path": meta.get("file_path", "")
            }
            for doc_id, meta in self._document_metadata.items()
        ]

    # =================================================

    def search_parallel(self, document_ids, query_embedding, top_k=5):
        if not document_ids or document_ids == ["all"]:
            document_ids = list(self._document_metadata.keys())

        futures = [
            self._executor.submit(self.search, doc_id, query_embedding, top_k)
            for doc_id in document_ids
            if doc_id in self._document_metadata
        ]

        all_results = []

        for f in concurrent.futures.as_completed(futures):
            all_results.extend(f.result())

        seen = set()
        unique = []

        for r in all_results:
            if r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                unique.append(r)

        unique.sort(key=lambda x: x["similarity_score"], reverse=True)

        return unique[:top_k]

    # =================================================

    def document_exists(self, document_id):

        return document_id in self._document_metadata

    def get_document_metadata(self, document_id):

        return self._document_metadata.get(document_id)

    def delete_document(self, document_id):

        if document_id not in self._document_metadata:
            return False

        collection_name = "documents"
        
        try:
             collection = self.client.get_collection(collection_name)
             collection.delete(where={"document_id": document_id})
        except Exception:
            pass

        del self._document_metadata[document_id]
        self._save_metadata()

        return True



_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

