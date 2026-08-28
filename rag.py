"""
RAG (Retrieval-Augmented Generation) Module — User-Scoped FAISS
Each authenticated user gets their own FAISS vector store so that
documents from different users are never mixed.
"""
import os
import shutil
from typing import List, Tuple, Dict, Any, Optional
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import numpy as np
import pickle

# ---------------------------
# GLOBALS
# ---------------------------
embeddings_model = None


# ---------------------------
# USER-SCOPED DB PATH
# ---------------------------
def get_user_db_path(base_path: str, user_id: int) -> str:
    """Return the FAISS index directory for a specific user."""
    return os.path.join(base_path, f"user_{user_id}")


# ---------------------------
# VECTOR STORE
# ---------------------------
class FAISSVectorStore:
    """
    Wraps a FAISS index with document chunks and optional source metadata.

    documents: list of raw text chunks
    sources:   list of dicts with keys: filename, page (parallel to documents)
    """
    def __init__(self, embeddings, documents, index, sources=None):
        self.embeddings = embeddings
        self.documents = documents
        self.index = index
        self.sources = sources or [{"filename": "unknown", "page": None}] * len(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Returns list of dicts: {chunk, filename, page}"""
        print(f"[DOC]  Searching {len(self.documents)} chunks for relevant content...")
        query_embedding = embeddings_model.encode([query])[0].astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), k)

        results = []
        for i in indices[0]:
            if i < len(self.documents):
                results.append({
                    "chunk": self.documents[i],
                    "filename": self.sources[i].get("filename", "document"),
                    "page": self.sources[i].get("page"),
                })
        return results


# ---------------------------
# EMBEDDINGS
# ---------------------------
def initialize_embeddings():
    global embeddings_model
    if embeddings_model is None:
        print("[DOC]  Loading embedding model (first time — may take ~30s)...")
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[DOC]  Embedding model ready")
    return embeddings_model


# ---------------------------
# STEP 1: LOAD PDF
# ---------------------------
def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Load PDF pages.
    Returns list of {text, page} dicts, one per page.
    """
    print(f"[DOC]  Step 1: Loading PDF from {pdf_path}")
    try:
        pdf_reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"text": text, "page": i})

        print(f"[DOC]  Loaded {len(pages)} pages with text")
        return pages

    except Exception as e:
        print(f"[DOC]  PDF Error: {e}")
        return []


# ---------------------------
# STEP 2: SPLIT TEXT
# ---------------------------
def chunk_text(pages: List[Dict[str, Any]], filename: str) -> Tuple[List[str], List[Dict]]:
    """
    Split page texts into overlapping chunks.
    Returns (chunks, sources) where sources[i] = {filename, page}.
    """
    print(f"[DOC]  Step 2: Splitting text into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    all_chunks = []
    all_sources = []

    for page_data in pages:
        page_chunks = splitter.split_text(page_data["text"])
        for chunk in page_chunks:
            all_chunks.append(chunk)
            all_sources.append({
                "filename": filename,
                "page": page_data["page"]
            })

    print(f"[DOC]  Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks, all_sources


# ---------------------------
# STEP 3: CREATE VECTOR DB
# ---------------------------
def create_vector_database(
    chunks: List[str],
    sources: List[Dict],
    db_path: str = "./faiss_index"
) -> "FAISSVectorStore":
    """Create a FAISS index from chunk embeddings. Saves index + documents + sources."""
    print(f"[DOC]  Step 3: Generating embeddings for {len(chunks)} chunks...")

    model = initialize_embeddings()

    embeddings = []
    for i, doc in enumerate(chunks):
        if i % 20 == 0 and i > 0:
            print(f"[DOC]    Processed {i}/{len(chunks)} chunks")
        embeddings.append(model.encode(doc))

    embeddings = np.array(embeddings).astype("float32")

    print(f"[DOC]  Building FAISS index (dim={embeddings.shape[1]})...")
    index = IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(db_path, exist_ok=True)

    import faiss
    faiss.write_index(index, os.path.join(db_path, "index.faiss"))

    with open(os.path.join(db_path, "documents.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    with open(os.path.join(db_path, "sources.pkl"), "wb") as f:
        pickle.dump(sources, f)

    print(f"[DOC]  Vector DB saved to {db_path}")
    return FAISSVectorStore(model, chunks, index, sources)


# ---------------------------
# LOAD EXISTING DB
# ---------------------------
def load_vector_database(db_path: str = "./faiss_index") -> "FAISSVectorStore | None":
    """Load a previously saved FAISS index from disk."""
    print(f"[DOC]  Loading vector DB from {db_path}...")

    try:
        model = initialize_embeddings()

        import faiss
        index = faiss.read_index(os.path.join(db_path, "index.faiss"))

        with open(os.path.join(db_path, "documents.pkl"), "rb") as f:
            documents = pickle.load(f)

        # Load sources if available (backward-compatible)
        sources_path = os.path.join(db_path, "sources.pkl")
        if os.path.exists(sources_path):
            with open(sources_path, "rb") as f:
                sources = pickle.load(f)
        else:
            sources = [{"filename": "document", "page": None}] * len(documents)

        print(f"[DOC]  Vector DB loaded: {len(documents)} chunks")
        return FAISSVectorStore(model, documents, index, sources)

    except Exception as e:
        print(f"[DOC]  Load Error: {e}")
        return None


# ---------------------------
# RETRIEVE
# ---------------------------
def retrieve_context(query: str, vector_store: FAISSVectorStore, k: int = 4) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a query.
    For summary-type queries, sample evenly across the document.
    Returns list of {chunk, filename, page} dicts.
    """
    query_lower = query.lower().strip()
    # Broad set of phrases that mean "give me an overview / extract everything"
    summary_keywords = [
        'summarize', 'summary', 'about', 'what is in', 'tell me about',
        'overview', 'main topic', 'content of', 'what is this',
        'what did i upload', 'uploaded pdf', 'tell me', 'what are',
        'key points', 'highlights', 'projects', 'experience', 'skills',
        'what does it say', 'what does this', 'describe', 'explain this',
        'give me', 'show me', 'list', 'mention', 'this pdf', 'this document',
        'this file', 'the pdf', 'the document', 'the file',
    ]

    if any(kw in query_lower for kw in summary_keywords):
        total = len(vector_store.documents)
        if total <= 12:
            indices = list(range(total))
        else:
            step = max(1, total // 12)
            indices = [min(i * step, total - 1) for i in range(12)]
        return [
            {
                "chunk": vector_store.documents[i],
                "filename": vector_store.sources[i].get("filename", "document"),
                "page": vector_store.sources[i].get("page"),
            }
            for i in indices
        ]

    return vector_store.similarity_search(query, k)


# ---------------------------
# GENERATE CONTEXT + SOURCES
def _clean_display_filename(name: str) -> str:
    """Strip any internal user_id + uuid prefix from stored filenames."""
    if not name:
        return "document"
    import re
    # Matches patterns like '5_c07d32737d6b4f4fb595bec91acfaf26_filename.pdf'
    cleaned = re.sub(r'^\d+_[0-9a-fA-F]{16,36}_', '', name)
    return cleaned if cleaned else name


# ---------------------------
# GENERATE CONTEXT + SOURCES
# ---------------------------
def generate_rag_context(
    query: str,
    vector_store: FAISSVectorStore,
    k: int = 6
) -> Tuple[str, List[Dict]]:
    """
    Retrieve relevant chunks and build an LLM context string.

    Returns:
        context (str)  — formatted text for the LLM prompt
        doc_sources    — list of unique source refs [{filename, pages: [n, ...]}]
    """
    result_chunks = retrieve_context(query, vector_store, k)

    if not result_chunks:
        return "No context found in the uploaded documents.", []

    # Build LLM context
    context_parts = []
    for r in result_chunks:
        clean_name = _clean_display_filename(r.get("filename", "document"))
        page_info = f" (page {r['page']})" if r.get("page") else ""
        context_parts.append(f"[From: {clean_name}{page_info}]\n{r['chunk']}")

    context = "\n\n".join(context_parts)

    # Build deduplicated source list for the frontend
    source_map: Dict[str, set] = {}
    for r in result_chunks:
        fname = _clean_display_filename(r.get("filename", "document"))
        page = r.get("page")
        if fname not in source_map:
            source_map[fname] = set()
        if page is not None:
            source_map[fname].add(page)

    doc_sources = [
        {"filename": fname, "pages": sorted(pages)}
        for fname, pages in source_map.items()
    ]

    print(f"[DOC]  Relevant chunks found: {len(result_chunks)}")
    print(f"[DOC]  Sources: {doc_sources}")
    return context, doc_sources


# ---------------------------
# MAIN PIPELINE
# ---------------------------
def process_pdf(
    pdf_path: str,
    db_path: str = "./faiss_index",
    user_id: int = 0,
    original_filename: Optional[str] = None
) -> Tuple[bool, str, int]:
    """
    Full RAG pipeline for a single PDF.
    Stores the FAISS index in a user-scoped subdirectory.

    Returns: (success, message, chunk_count)
    """
    print(f"\n[DOC]  ===== STARTING RAG PIPELINE for user {user_id} =====")

    display_filename = original_filename or _clean_display_filename(os.path.basename(pdf_path))

    # Step 1: Load PDF with page info
    pages = load_pdf(pdf_path)
    if not pages:
        return False, "Failed to extract text from PDF", 0

    # Step 2: Chunk with source tracking
    chunks, sources = chunk_text(pages, display_filename)
    if not chunks:
        return False, "Failed to create chunks", 0

    print(f"[DOC]  Chunks created: {len(chunks)}")

    # Step 3: Create user-scoped vector DB
    user_db_path = get_user_db_path(db_path, user_id)
    db = create_vector_database(chunks, sources, user_db_path)
    if db is None:
        return False, "Failed to create embeddings", 0

    print(f"[DOC]  ===== RAG PIPELINE COMPLETE for user {user_id} =====\n")
    return True, "Success", len(chunks)


# ---------------------------
# DB UTILITIES
# ---------------------------
def check_database_exists(db_path: str = "./faiss_index", user_id: int = 0) -> bool:
    """Check if a FAISS index exists for the given user."""
    user_db_path = get_user_db_path(db_path, user_id)
    return (
        os.path.exists(os.path.join(user_db_path, "index.faiss")) and
        os.path.exists(os.path.join(user_db_path, "documents.pkl"))
    )


def delete_vector_database(db_path: str = "./faiss_index", user_id: int = 0) -> Tuple[bool, str]:
    """Delete the FAISS index for a specific user."""
    user_db_path = get_user_db_path(db_path, user_id)
    try:
        if os.path.exists(user_db_path):
            shutil.rmtree(user_db_path)
            return True, "Deleted"
        return False, "Not found"
    except Exception as e:
        return False, str(e)
