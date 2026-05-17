"""
RAG Engine — Retrieval-Augmented Generation for KeratoScan AI.

Reads documents from the 'RAG Files' directory, chunks them, builds a TF-IDF index,
and retrieves the most relevant passages to augment the Gemini prompt.

No external vector DB required — runs fully locally with scikit-learn.
"""

import os
import re
import math
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
RAG_DIR = Path(__file__).parent.parent / "RAG Files"
CHUNK_SIZE = 400          # tokens (approx words) per chunk
CHUNK_OVERLAP = 80        # overlapping words between chunks
TOP_K = 5                 # number of chunks to retrieve
MIN_SCORE = 0.05          # minimum TF-IDF cosine similarity to include a chunk
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


# ── Data structures ───────────────────────────────────────────────────────────

class DocumentChunk:
    """A piece of a RAG document with metadata."""
    def __init__(self, text: str, source: str, chunk_id: int):
        self.text = text
        self.source = source          # filename (without path)
        self.chunk_id = chunk_id
        self.uid = hashlib.md5(f"{source}:{chunk_id}".encode()).hexdigest()[:8]

    def __repr__(self):
        return f"<Chunk {self.uid} | {self.source} | {len(self.text)} chars>"


# ── Document loading ──────────────────────────────────────────────────────────

def _load_txt(path: Path) -> str:
    """Load a plain text or markdown file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    """Load a PDF file using pypdf (optional dependency)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        return ""
    except Exception:
        return ""


def load_rag_documents(rag_dir: Path = RAG_DIR) -> List[Tuple[str, str]]:
    """
    Load all supported documents from rag_dir.
    Returns list of (filename, full_text) tuples.
    """
    docs = []
    if not rag_dir.exists():
        return docs

    for fp in sorted(rag_dir.iterdir()):
        if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if fp.suffix.lower() == ".pdf":
                text = _load_pdf(fp)
            else:
                text = _load_txt(fp)
            if text.strip():
                docs.append((fp.name, text))
        except Exception:
            continue
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Simple whitespace tokeniser."""
    return text.split()


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """Split text into overlapping word-based chunks."""
    words = _tokenise(text)
    chunks: List[DocumentChunk] = []
    step = max(1, chunk_size - overlap)
    for i, start in enumerate(range(0, len(words), step)):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) < 20:        # skip tiny trailing chunks
            break
        chunks.append(DocumentChunk(" ".join(chunk_words), source, i))
    return chunks


# ── TF-IDF Index ──────────────────────────────────────────────────────────────

class TFIDFIndex:
    """
    Lightweight TF-IDF retrieval index built with scikit-learn.
    Cached in Streamlit session state to avoid re-indexing on every run.
    """

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self._vectorizer = None
        self._matrix = None
        self._built = False

    def build(self):
        """Build the TF-IDF matrix from all chunks."""
        if not self.chunks:
            self._built = False
            return self
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize as sk_normalize
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                stop_words="english",
            )
            corpus = [c.text for c in self.chunks]
            self._matrix = self._vectorizer.fit_transform(corpus)
            self._matrix = sk_normalize(self._matrix)  # L2 normalise for cosine sim
            self._built = True
        except ImportError:
            # Graceful fallback — build a simple bag-of-words keyword index
            self._built = self._build_fallback()
        except Exception as e:
            self._built = False
        return self

    def _build_fallback(self) -> bool:
        """Pure-Python keyword frequency fallback (no sklearn required)."""
        import math, re
        stop = {
            "a","an","the","is","are","was","were","of","in","to",
            "and","or","for","with","on","at","by","from","as",
        }
        def tokenise(t):
            return [w.lower() for w in re.findall(r"[a-z]+", t.lower()) if w not in stop and len(w) > 2]
        # Build DF
        df = {}
        corpus_tokens = []
        for c in self.chunks:
            toks = set(tokenise(c.text))
            corpus_tokens.append(list(toks))
            for t in toks:
                df[t] = df.get(t, 0) + 1
        N = len(self.chunks)
        # Store idf
        self._idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
        self._corpus_tokens = corpus_tokens
        self._fallback_mode = True
        return True

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve top-k chunks most relevant to query_text.
        Returns list of (chunk, cosine_similarity_score) sorted descending.
        """
        if not self._built or not self.chunks:
            return []

        # Fallback pure-Python mode
        if getattr(self, "_fallback_mode", False):
            return self._query_fallback(query_text, top_k, min_score)

        try:
            from sklearn.preprocessing import normalize as sk_normalize
            import numpy as np

            q_vec = self._vectorizer.transform([query_text])
            q_vec = sk_normalize(q_vec)
            scores = (self._matrix @ q_vec.T).toarray().flatten()
            top_indices = scores.argsort()[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score >= min_score:
                    results.append((self.chunks[idx], score))
            return results
        except Exception:
            return self._query_fallback(query_text, top_k, min_score)

    def _query_fallback(
        self,
        query_text: str,
        top_k: int,
        min_score: float,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Pure-Python BM25-style keyword matching fallback."""
        import re, math
        stop = {
            "a","an","the","is","are","was","were","of","in","to",
            "and","or","for","with","on","at","by","from","as",
        }
        def tokenise(t):
            return [w.lower() for w in re.findall(r"[a-z]+", t.lower()) if w not in stop and len(w) > 2]

        q_tokens = tokenise(query_text)
        if not q_tokens:
            return []

        idf = getattr(self, "_idf", {})
        corpus_tokens = getattr(self, "_corpus_tokens", [])

        scored = []
        for i, (chunk, toks) in enumerate(zip(self.chunks, corpus_tokens)):
            tok_set = set(toks)
            score = sum(idf.get(t, 0) for t in q_tokens if t in tok_set)
            # Normalise by chunk length
            score = score / (math.log(len(toks) + 2))
            scored.append((chunk, score))

        scored.sort(key=lambda x: -x[1])
        max_score = scored[0][1] if scored else 1.0
        if max_score == 0:
            return []
        # Normalise scores to [0, 1]
        results = []
        for chunk, score in scored[:top_k]:
            norm_score = score / max_score
            if norm_score >= min_score:
                results.append((chunk, norm_score))
        return results

    @property
    def is_ready(self) -> bool:
        return self._built

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_documents(self) -> int:
        return len(set(c.source for c in self.chunks))


# ── Session-cached index builder ──────────────────────────────────────────────

def get_rag_index(rag_dir: Path = RAG_DIR) -> TFIDFIndex:
    """
    Build (or return cached) the RAG index.
    Uses a content hash to detect document changes and rebuild if needed.
    """
    # Compute a hash of all document filenames + sizes to detect changes
    doc_fingerprint = _compute_dir_fingerprint(rag_dir)

    # Use cached index if fingerprint unchanged
    cached = st.session_state.get("_rag_index")
    cached_fp = st.session_state.get("_rag_index_fingerprint")

    if cached is not None and cached_fp == doc_fingerprint and cached.is_ready:
        return cached

    # Build fresh index
    docs = load_rag_documents(rag_dir)
    all_chunks: List[DocumentChunk] = []
    for filename, text in docs:
        all_chunks.extend(chunk_text(text, filename))

    index = TFIDFIndex(all_chunks).build()

    # Cache in session state
    st.session_state["_rag_index"] = index
    st.session_state["_rag_index_fingerprint"] = doc_fingerprint

    return index


def _compute_dir_fingerprint(rag_dir: Path) -> str:
    """Compute a hash based on filenames and sizes in rag_dir."""
    if not rag_dir.exists():
        return "empty"
    entries = []
    for fp in sorted(rag_dir.iterdir()):
        if fp.suffix.lower() in SUPPORTED_EXTENSIONS:
            entries.append(f"{fp.name}:{fp.stat().st_size}")
    return hashlib.md5("|".join(entries).encode()).hexdigest()


# ── Context builder ───────────────────────────────────────────────────────────

def build_rag_context(
    query: str,
    index: TFIDFIndex,
    top_k: int = TOP_K,
) -> Tuple[str, List[Dict]]:
    """
    Retrieve relevant chunks and format them as a context block for the LLM prompt.

    Returns:
        context_text: Formatted string to inject into the prompt
        sources: List of dicts with source metadata for display
    """
    if not index.is_ready:
        return "", []

    results = index.query(query, top_k=top_k)
    if not results:
        return "", []

    context_parts = []
    sources = []

    for i, (chunk, score) in enumerate(results, 1):
        context_parts.append(
            f"[Reference {i} | Source: {chunk.source} | Relevance: {score:.2f}]\n"
            f"{chunk.text}"
        )
        sources.append({
            "index": i,
            "source": chunk.source,
            "score": score,
            "preview": chunk.text[:180].replace("\n", " ") + "…",
        })

    context_text = (
        "## Retrieved Clinical Knowledge Base\n\n"
        "The following passages from the medical literature are relevant to this case:\n\n"
        + "\n\n---\n\n".join(context_parts)
    )
    return context_text, sources


def build_query_from_results(
    ensemble_result: Dict,
    image_result: Dict,
    tabular_result: Dict,
    feature_importance: Optional[Dict] = None,
    modality_weights: Optional[Dict] = None,
) -> str:
    """
    Build a rich query string from model outputs to drive RAG retrieval.
    More specific query → better chunk retrieval.
    """
    parts = []

    severity = ensemble_result.get("severity", "")
    diagnosis = ensemble_result.get("predicted_label", "")
    kc_prob = ensemble_result.get("keratoconus_prob", 0.0)

    parts.append(f"keratoconus {diagnosis} {severity}")

    if kc_prob > 0.65:
        parts.append("management treatment contact lens cross-linking CXL surgical")
    elif kc_prob > 0.4:
        parts.append("mild moderate keratoconus monitoring progression")
    elif kc_prob > 0.2:
        parts.append("subclinical forme fruste keratoconus screening surveillance")
    else:
        parts.append("normal cornea no keratoconus observation")

    # Add top modalities
    if modality_weights:
        top_mods = sorted(modality_weights.items(), key=lambda x: -x[1])[:3]
        for mod, _ in top_mods:
            mod_map = {
                "CT_A": "corneal thickness pachymetry thinning",
                "EC_A": "epithelial thickness donut pattern",
                "EC_P": "epithelial posterior curvature Bowman",
                "Elv_A": "anterior elevation best fit sphere island",
                "Elv_P": "posterior elevation ectasia protrusion",
                "Sag_A": "sagittal curvature Kmax steepening astigmatism",
                "Sag_P": "posterior sagittal curvature",
            }
            parts.append(mod_map.get(mod, mod))

    # Add top features
    if feature_importance:
        top_feats = sorted(feature_importance.items(), key=lambda x: -x[1])[:3]
        for feat, _ in top_feats:
            parts.append(feat.replace("_", " "))

    return " ".join(parts)


# ── Index status helper ───────────────────────────────────────────────────────

def get_index_stats(index: TFIDFIndex) -> Dict:
    """Return a summary dict for display in the UI."""
    return {
        "ready": index.is_ready,
        "num_documents": index.num_documents,
        "num_chunks": index.num_chunks,
        "rag_dir": str(RAG_DIR),
        "supported_formats": list(SUPPORTED_EXTENSIONS),
    }
