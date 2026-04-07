#!/usr/bin/env python3
"""
RAG CLI for Saarland University Module Handbooks
Embeddings : sentence-transformers (local, free)
Vector DB  : ChromaDB (local, free)
LLM        : Ollama (local, free)
PDF Parser : pdfplumber (local, free)
"""

import os, sys, re, argparse, textwrap, requests, json, logging, sqlite3
from pathlib import Path
from typing import List, Dict, Optional

import pdfplumber
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ── CONFIG ────────────────────────────────────────────────────────────────────
EMBED_MODEL      = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_URL       = os.getenv("OLLAMA_URL", "http://localhost:11434")
DB_DIR           = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K    = int(os.getenv("DEFAULT_TOP_K", "6"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))
MAX_CONTEXT_CHARS= int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
COLLECTION       = "module_handbooks"
INDEX_DB         = os.path.join(DB_DIR, "module_index.db")

OLLAMA_MODELS = [
    "llama3.2",
    "mistral",
    "llama3.2:1b",
    "phi3",
    "gemma2:2b",
]

# ── PRETTY PRINT HELPERS ──────────────────────────────────────────────────────

def _hr(char="─", width=60):
    return char * width

def _box(title: str, width=60):
    pad = width - 2 - len(title)
    left = pad // 2
    right = pad - left
    return f"┌{'─'*width}┐\n│{' '*left}{title}{' '*right}│\n└{'─'*width}┘"

def print_info(msg: str):    print(f"  ℹ  {msg}")
def print_ok(msg: str):      print(f"  ✓  {msg}")
def print_warn(msg: str):    print(f"  ⚠  {msg}")
def print_error(msg: str):   print(f"  ✗  {msg}")
def print_hr():              print(f"  {_hr()}")

# ── PDF + CHUNKING ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF, returning page-annotated text."""
    parts = []
    pdf_path = str(pdf_path)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        parts.append(f"\n--- Page {i+1} ---\n{text}")
                except Exception as e:
                    logger.debug(f"Page {i+1} extraction failed: {e}")
    except pdfplumber.PDFException as e:
        raise ValueError(f"PDF parse error in '{pdf_path}': {e}")
    except FileNotFoundError:
        raise ValueError(f"File not found: '{pdf_path}'")
    except Exception as e:
        raise ValueError(f"Cannot open PDF '{pdf_path}': {type(e).__name__}: {e}")
    if not parts:
        raise ValueError(f"No text extracted from '{pdf_path}' (corrupted or image-only PDF?)")
    return "\n".join(parts)


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split text into overlapping chunks on paragraph boundaries."""
    chunks = []
    text = re.sub(r'\n{3,}', '\n\n', text)
    paragraphs = re.split(r'(?<=\n\n)', text)
    current = ""
    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += para
        else:
            if current.strip():
                chunks.append({"text": current.strip(), "source": Path(source).name, "chunk_id": len(chunks)})
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + para
    if current.strip():
        chunks.append({"text": current.strip(), "source": Path(source).name, "chunk_id": len(chunks)})
    return chunks

# ── DB + EMBEDDINGS ───────────────────────────────────────────────────────────

def get_collection():
    """Get or create ChromaDB collection."""
    try:
        Path(DB_DIR).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        return client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}")
        raise

# FIX 1: Singleton — model is loaded once per process, not per call
_embed_model: Optional[SentenceTransformer] = None

def get_embed_model() -> SentenceTransformer:
    """Load and cache the embedding model (loaded only once per process)."""
    global _embed_model
    if _embed_model is None:
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _embed_model

# ── STRUCTURED MODULE INDEX ───────────────────────────────────────────────────

def _fix_spaces(text: str) -> str:
    """Re-insert spaces lost in PDF extraction (e.g. 'ComputerGraphics' → 'Computer Graphics')."""
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

def get_index_db() -> sqlite3.Connection:
    """Open (or create) the SQLite module index."""
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(INDEX_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS modules (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            name   TEXT,
            code   TEXT,
            ects   INTEGER,
            sws    INTEGER,
            source TEXT,
            body   TEXT
        )
    """)
    con.commit()
    return con

def index_modules_from_text(raw_text: str, source: str, con: sqlite3.Connection) -> int:
    """
    Parse module metadata using two complementary strategies:

    Strategy A — "name + code on one line, stats table follows":
        Matches lines like:  Computer Graphics CG
        Then scans forward for the stats row containing SWS / ECTS numbers.

    Strategy B — "SWS ECTS header row" (original heuristic, kept as fallback):
        Looks for the literal header row and reads the row above it as the name.

    Both strategies use _fix_spaces() to recover camelCase merges from pdfplumber.
    """
    # Split on page markers
    page_blocks = re.split(r'---\s*Page \d+\s*---', raw_text)
    inserted = 0

    # Regex: "One or more Title-Case words  UPPERCASE-CODE" at end of line
    # e.g. "Computer Graphics CG" or "High-Level Computer Vision HLCV"
    name_code_re = re.compile(
        r'^(?P<name>[A-Z][A-Za-z0-9 ,\-&]+?)\s+(?P<code>[A-Z]{2,8})\s*$'
    )
    # Matches a data row that has exactly two integers that could be SWS and ECTS
    # e.g. "1-3  4  at least every two years  1 semester  6  9"
    stats_re = re.compile(r'\b(\d{1,2})\s+(\d{1,2})\s*$')

    seen_names = set()

    for block in page_blocks:
        text  = _fix_spaces(block)
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # ── Strategy A ────────────────────────────────────────────────────────
        for j, line in enumerate(lines):
            m = name_code_re.match(line)
            if not m:
                continue
            name = m.group("name").strip()
            code = m.group("code").strip()

            # Sanity: name must have at least two words or be > 6 chars
            if len(name) < 5 or name in seen_names:
                continue

            # Skip obvious table-header false positives
            if name.lower() in ("st semester std st sem cycle duration", "module category"):
                continue

            # Scan forward up to 8 lines for the stats row
            sws, ects = None, None
            body_lines = []
            for k in range(j + 1, min(j + 9, len(lines))):
                sm = stats_re.search(lines[k])
                if sm:
                    sws  = int(sm.group(1))
                    ects = int(sm.group(2))
                    body_lines = lines[k:]
                    break

            if ects is None:
                continue   # couldn't confirm this is a module header

            seen_names.add(name)
            body = "\n".join(body_lines[:30])
            con.execute(
                "INSERT INTO modules (name, code, ects, sws, source, body) VALUES (?,?,?,?,?,?)",
                (name, code, ects, sws, source, body)
            )
            inserted += 1

        # ── Strategy B (fallback) ─────────────────────────────────────────────
        for j, line in enumerate(lines):
            if not re.search(r'SWS\s+ECTS', line):
                continue
            if j == 0 or j + 1 >= len(lines):
                continue
            name_line = lines[j - 1]
            if name_line in seen_names:
                continue

            vals_line = lines[j + 1]
            nums = re.findall(r'\b(\d+)\b', vals_line)
            sws  = int(nums[-2]) if len(nums) >= 2 else None
            ects = int(nums[-1]) if len(nums) >= 1 else None

            code_m = re.search(r'\s+([A-Z]{2,8})\s*$', name_line)
            code   = code_m.group(1).strip() if code_m else None
            clean  = re.sub(r'\s+[A-Z]{2,8}\s*$', '', name_line).strip()
            if not clean or clean in seen_names:
                continue

            seen_names.add(clean)
            body = "\n".join(lines[j:j+30])
            con.execute(
                "INSERT INTO modules (name, code, ects, sws, source, body) VALUES (?,?,?,?,?,?)",
                (clean, code, ects, sws, source, body)
            )
            inserted += 1

    con.commit()
    return inserted


def delete_index_for_source(source: str, con: sqlite3.Connection):
    con.execute("DELETE FROM modules WHERE source=?", (source,))
    con.commit()

def is_listing_query(q: str) -> bool:
    q_lower = q.lower()
    listing_phrases = [
        "list", "list all", "list out", "show all", "which modules",
        "what modules", "all modules", "modules with", "modules that have",
        "how many modules", "count modules"
    ]
    return any(p in q_lower for p in listing_phrases)

def filter_query(user_query: str, con: sqlite3.Connection) -> Optional[str]:
    """Handle structured filter queries directly from the SQLite index."""
    q = user_query.lower()

    ects_m = re.search(r'(\d+)\s*ects', q, re.IGNORECASE)
    sws_m  = re.search(r'(\d+)\s*sws',  q, re.IGNORECASE)

    rows  = None
    label = ""

    if ects_m:
        val   = int(ects_m.group(1))
        rows  = con.execute(
            "SELECT name, code, ects, sws, source FROM modules WHERE ects=? ORDER BY source, name",
            (val,)
        ).fetchall()
        label = f"{val} ECTS"
    elif sws_m:
        val   = int(sws_m.group(1))
        rows  = con.execute(
            "SELECT name, code, ects, sws, source FROM modules WHERE sws=? ORDER BY source, name",
            (val,)
        ).fetchall()
        label = f"{val} SWS"
    elif any(p in q for p in ("list", "all modules", "show all")):
        rows  = con.execute(
            "SELECT name, code, ects, sws, source FROM modules ORDER BY source, name"
        ).fetchall()
        label = "all modules"
    else:
        return None

    if not rows:
        return "  No modules found matching your query in the index."

    lines = [f"\n  Modules — {label} ({len(rows)} found)\n  {_hr()}"]
    current_src = None
    for name, code, ects, sws, source in rows:
        if source != current_src:
            lines.append(f"\n  📄 {source}")
            current_src = source
        code_str = f" [{code}]" if code else ""
        ects_str = f"  {ects} ECTS" if ects else ""
        sws_str  = f"  {sws} SWS"  if sws  else ""
        lines.append(f"    • {name}{code_str}{ects_str}{sws_str}")
    return "\n".join(lines) + "\n"

# ── INGEST ────────────────────────────────────────────────────────────────────

def ingest(pdf_paths: List[str], force_reindex: bool = False):
    """Ingest PDFs into the vector database and structured module index."""
    print(f"\n  {_hr()}")
    print(f"  Ingest")
    print(f"  {_hr()}")

    try:
        col   = get_collection()
        model = get_embed_model()
    except Exception as e:
        print_error(f"Cannot initialize database/model: {e}")
        return

    # Determine which files to process
    try:
        existing_docs = col.get(include=["metadatas"])
        existing = {m["source"] for m in existing_docs.get("metadatas", [])}
    except Exception:
        existing = set()

    new_files = []
    for p in pdf_paths:
        if not Path(p).exists():
            print_warn(f"File not found: {p}")
            continue
        fname = Path(p).name
        if fname in existing and not force_reindex:
            print_info(f"Skipping {fname} (already indexed — use --force-reindex to overwrite)")
            continue
        if fname in existing and force_reindex:
            try:
                col.delete(where={"source": fname})
                print_info(f"Removed old index for {fname}")
            except Exception as e:
                print_warn(f"Could not delete old chunks for {fname}: {e}")
        new_files.append(p)

    if not new_files:
        print_info("Nothing new to ingest.")
        print()
        return

    # FIX 2: Extract PDF text ONCE and reuse for both chunking and module indexing
    extracted: Dict[str, str] = {}
    ids, texts, metas = [], [], []

    for path in new_files:
        fname = Path(path).name
        try:
            pdf_text = extract_text_from_pdf(path)   # ← single extraction
            extracted[fname] = pdf_text
            chunks = chunk_text(pdf_text, path)

            for c in chunks:
                # FIX 3: Use fname directly so IDs are stable and unique per file
                chunk_id = f"{fname}__chunk{c['chunk_id']}"
                ids.append(chunk_id)
                texts.append(c["text"])
                metas.append({"source": fname})

            print_ok(f"Extracted  {fname}  ({len(chunks)} chunks)")
        except ValueError as e:
            print_error(str(e))
        except Exception as e:
            print_error(f"Unexpected error processing {path}: {type(e).__name__}: {e}")

    if not texts:
        print_error("No text extracted from any files.")
        return

    # Embed and store
    try:
        print_info(f"Embedding {len(texts)} chunks …")
        embs = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch      = texts[i:i + EMBED_BATCH_SIZE]
            batch_embs = model.encode(batch, show_progress_bar=False)
            embs.extend(batch_embs.tolist())
        col.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
        print_ok(f"Vector DB updated  (total chunks: {col.count()})")
    except Exception as e:
        print_error(f"Failed to embed/store chunks: {e}")
        return

    # Build structured module index — reuse already-extracted text
    try:
        idx_con = get_index_db()
        for path in new_files:
            fname = Path(path).name
            if fname not in extracted:
                continue
            try:
                delete_index_for_source(fname, idx_con)
                n = index_modules_from_text(extracted[fname], fname, idx_con)
                if n:
                    print_ok(f"Module index  {fname}  ({n} modules parsed)")
                else:
                    print_warn(f"Module index  {fname}  (0 modules parsed — layout may differ)")
            except Exception as e:
                print_warn(f"Could not build module index for {fname}: {e}")
        idx_con.close()
    except Exception as e:
        print_warn(f"Module index unavailable: {e}")

    print()

# ── RETRIEVE ──────────────────────────────────────────────────────────────────

def retrieve(user_query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """Retrieve relevant chunks from ChromaDB."""
    try:
        col = get_collection()
        if col.count() == 0:
            return []
        model = get_embed_model()          # ← cached, no reload
        q_emb = model.encode([user_query]).tolist()
        res   = col.query(
            query_embeddings=q_emb,
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"]
        )
        results = []
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0]
        ):
            results.append({
                "text":   doc,
                "source": meta.get("source", "unknown"),
                "score":  round(1 - dist, 4)
            })
        return results
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []

# ── OLLAMA API ────────────────────────────────────────────────────────────────

def ollama_is_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def ollama_list_models() -> List[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"].split(":")[0] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []

def ollama_find_model(model_list: Optional[List[str]] = None) -> Optional[str]:
    if model_list is None:
        model_list = OLLAMA_MODELS
    pulled = ollama_list_models()
    for m in model_list:
        base = m.split(":")[0]
        if base in pulled or m in pulled:
            return m
    return None

def ask_ollama(user_query: str, hits: List[Dict], model_name: str) -> str:
    """Stream a response from Ollama with context."""
    ctx_parts  = []
    total_chars = 0
    for h in hits:
        part = f"[{h['source']}]\n{h['text']}"
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            break
        ctx_parts.append(part)
        total_chars += len(part)
    ctx = "\n\n".join(ctx_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise assistant for Saarland University students. "
                "Answer ONLY using the provided context. "
                "Quote exact values (ECTS, SWS, prerequisites, exam rules) directly from the context — do not guess or paraphrase numbers. "
                "If the answer is not explicitly in the context, say so. "
                "Do not invent module names, lecturers, or requirements."
            )
        },
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {user_query}"}
    ]

    payload = {
        "model":   model_name,
        "messages": messages,
        "stream":  True,
        "options": {"temperature": 0.2, "num_predict": 800, "top_k": 40, "top_p": 0.9}
    }

    full_text = []
    print(f"\n  Answer \n  {_hr()}")

    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120) as r:
            if r.status_code != 200:
                msg = f"[Error {r.status_code}: {r.text[:200]}]"
                print(f"  {msg}")
                return msg

            print("  ", end="", flush=True)
            col = 2
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    for char in token:
                        if char == "\n":
                            print()
                            print("  ", end="", flush=True)
                            col = 2
                        else:
                            print(char, end="", flush=True)
                            col += 1
                    full_text.append(token)
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    pass

    except requests.exceptions.ConnectionError:
        msg = "Cannot connect to Ollama. Is it running? Try: ollama serve"
        print_error(msg)
        return f"[Error: {msg}]"
    except requests.exceptions.Timeout:
        msg = "Ollama request timed out (inference took too long)"
        print_error(msg)
        return f"[Error: {msg}]"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print_error(msg)
        return f"[Error: {msg}]"

    print("\n")
    return "".join(full_text)

# ── QUERY ─────────────────────────────────────────────────────────────────────

def query(user_query: str, top_k: int = DEFAULT_TOP_K, show_sources: bool = True):
    """Execute a single query against the RAG pipeline."""
    if not user_query or not user_query.strip():
        print_error("Query cannot be empty.")
        return
    if not (1 <= top_k <= 100):
        print_error(f"--top-k must be between 1 and 100 (got {top_k}).")
        return

    user_query = user_query.strip()

    if not ollama_is_running():
        print_error("Ollama is not running.")
        print_info("Start it with:  ollama serve")
        return

    model_name = ollama_find_model()
    if not model_name:
        pulled = ollama_list_models()
        print_error("No suitable models found.")
        print_info(f"Pulled models: {', '.join(pulled) if pulled else 'none'}")
        print_info("Pull one with:  ollama pull llama3.2")
        return

    # Try structured index for listing/filter queries first
    if is_listing_query(user_query):
        try:
            idx_con = get_index_db()
            answer  = filter_query(user_query, idx_con)
            idx_con.close()
            if answer:
                print(answer)
                return
        except Exception as e:
            logger.warning(f"Index query failed, falling back to RAG: {e}")

    hits = retrieve(user_query, top_k)
    if not hits:
        print_error("No content found in database.")
        print_info("Run:  python rag.py ingest yourfile.pdf")
        return

    ask_ollama(user_query, hits, model_name)

    if show_sources:
        seen = set()
        lines = [f"  Sources\n  {_hr()}"]
        for h in hits:
            k = (h["source"], h["score"])
            if k not in seen:
                seen.add(k)
                bar_n  = int(h["score"] * 20)
                bar    = "█" * bar_n + "░" * (20 - bar_n)
                lines.append(f"  {h['source']:<42} {bar}  {h['score']:.3f}")
        print("\n".join(lines) + "\n")

# ── INTERACTIVE ───────────────────────────────────────────────────────────────

def interactive(top_k: int = DEFAULT_TOP_K):
    """Interactive Q&A session."""
    print(f"\n  {_hr()}")
    print(f"  Saarland University — Module Handbook Assistant")
    print(f"  {_hr()}")
    print(f"  Sample questions: Which modules have 6 ECTS? \n  What are the prerequisites for Computer Graphics? \n  List all modules with 4 SWS. \n  How many modules are there with 9 ECTS?")
    print(f"  {_hr()}")
    print(f"  Type 'quit' to exit, 'help' for commands.\n")

    while True:
        try:
            user_input = input("  You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        if user_input.lower() == "help":
            print(f"\n  Commands: quit, help")
            print(f"  Ask anything about the indexed modules.\n")
            continue

        query(user_input, top_k=top_k)

# ── STATUS ────────────────────────────────────────────────────────────────────

def status():
    """Show database and Ollama status."""
    print(f"\n  {_hr()}")
    print(f"  Status")
    print(f"  {_hr()}\n")

    # ChromaDB
    try:
        col   = get_collection()
        count = col.count()
        print(f"  ChromaDB")
        print(f"  {'Path':<14}: {Path(DB_DIR).absolute()}")
        print(f"  {'Collection':<14}: {COLLECTION}")
        print(f"  {'Chunks':<14}: {count}")

        if count > 0:
            meta    = col.get(include=["metadatas"])["metadatas"]
            sources = sorted({m["source"] for m in meta})
            print(f"  {'Files':<14}: {len(sources)}")
            for s in sources:
                n = sum(1 for m in meta if m["source"] == s)
                print(f"    • {s:<42} {n:>4} chunks")
    except Exception as e:
        print_error(f"Could not access ChromaDB: {e}")

    # Module index
    print()
    try:
        idx_con = get_index_db()
        total   = idx_con.execute("SELECT COUNT(*) FROM modules").fetchone()[0]
        print(f"  Module Index")
        print(f"  {'Path':<14}: {Path(INDEX_DB).absolute()}")
        print(f"  {'Modules':<14}: {total}")
        if total > 0:
            rows = idx_con.execute(
                "SELECT source, COUNT(*) FROM modules GROUP BY source ORDER BY source"
            ).fetchall()
            for src, n in rows:
                print(f"    • {src:<42} {n:>4} modules")
        idx_con.close()
    except Exception as e:
        print_warn(f"Module index unavailable: {e}")

    # Ollama
    print()
    print(f"  Ollama")
    is_running = ollama_is_running()
    print(f"  {'Running':<14}: {'Yes ✓' if is_running else 'No  (run: ollama serve)'}")
    if is_running:
        pulled = ollama_list_models()
        model  = ollama_find_model()
        print(f"  {'Models':<14}: {', '.join(pulled) if pulled else '(none pulled)'}")
        print(f"  {'Active':<14}: {model or '(none of preferred models pulled)'}")
    print()

# ── SETUP GUIDE ───────────────────────────────────────────────────────────────

def setup_guide():
    print(f"""
  {_hr()}
  Setup Guide
  {_hr()}

  1. Install Ollama
       https://ollama.com/download
       or:  brew install ollama

  2. Start Ollama
       ollama serve

  3. Pull a model
       ollama pull llama3.2       (recommended, ~2 GB)
       ollama pull llama3.2:1b   (fastest, ~1 GB)
       ollama pull mistral        (best quality, ~4 GB)

  4. Ingest your PDF
       python rag.py ingest handbook.pdf

  5. Start chatting
       python rag.py chat

  Environment variables (optional overrides):
    OLLAMA_URL        default: http://localhost:11434
    CHROMA_DB_PATH    default: ./chroma_db
    EMBED_MODEL       default: sentence-transformers/all-MiniLM-L6-v2
    CHUNK_SIZE        default: 800
    DEFAULT_TOP_K     default: 6
""")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG CLI — Saarland University Module Handbooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Commands:
          python rag.py setup                       Show setup guide
          python rag.py ingest handbook.pdf         Index PDF file(s)
          python rag.py ingest handbook.pdf --force-reindex
          python rag.py ask "Which modules cover cryptography?"
          python rag.py chat                        Interactive mode
          python rag.py status                      Show DB + Ollama info
        """)
    )
    parser.add_argument("--top-k",      type=int,  default=DEFAULT_TOP_K,
                        help=f"Chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--no-sources", action="store_true",
                        help="Hide source filenames in query output")
    parser.add_argument("--verbose",    action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--ollama-url", default=None,
                        help="Ollama endpoint (overrides OLLAMA_URL env)")
    parser.add_argument("--db-dir",     default=None,
                        help="ChromaDB directory (overrides CHROMA_DB_PATH env)")

    sub = parser.add_subparsers(dest="command")

    pi = sub.add_parser("ingest", help="Index PDF file(s)")
    pi.add_argument("pdfs", nargs="+")
    pi.add_argument("--force-reindex", action="store_true",
                    help="Delete and re-index even if already indexed")

    pa = sub.add_parser("ask",  help="Ask a single question")
    pa.add_argument("question", nargs="+")

    sub.add_parser("chat",   help="Interactive Q&A mode")
    sub.add_parser("status", help="Show DB + Ollama status")
    sub.add_parser("setup",  help="Show setup instructions")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("sentence_transformers").setLevel(logging.INFO)
        logging.getLogger("chromadb").setLevel(logging.WARNING)

    global OLLAMA_URL, DB_DIR, INDEX_DB
    if args.ollama_url:
        OLLAMA_URL = args.ollama_url
    if args.db_dir:
        DB_DIR  = args.db_dir
        INDEX_DB = os.path.join(DB_DIR, "module_index.db")

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "setup":
            setup_guide()
        elif args.command == "ingest":
            ingest(args.pdfs, force_reindex=getattr(args, "force_reindex", False))
        elif args.command == "ask":
            query(" ".join(args.question),
                  top_k=args.top_k,
                  show_sources=not args.no_sources)
        elif args.command == "chat":
            interactive(top_k=args.top_k)
        elif args.command == "status":
            status()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(130)
    except Exception as e:
        if args.verbose:
            logger.error(f"Fatal error: {e}", exc_info=True)
        else:
            print_error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()