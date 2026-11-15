#!/usr/bin/env python3
"""
RAG v2: accuracy-focused version of run_pdf.py.
Every improvement has an on/off switch (environment variable, 1 = on, 0 = off).

Usage: python rag_v2.py --question "..." [--rebuild] [--dump]
"""
import os
import re
import sys
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings


def flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default) == "1"


# ---------------- SETTINGS (change with environment variables) ----------------
DEFAULT_PDF = "major_handbook_pdfs/CS_MajorHandbook.pdf"
VECTOR_ROOT = "chroma_db"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
NUM_CTX = int(os.getenv("NUM_CTX", "4096"))
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
TOP_K = int(os.getenv("TOP_K", "6"))
CANDIDATES = int(os.getenv("CANDIDATES", "15"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Switches that change the INDEX (changing them builds a new index automatically)
USE_CLEAN = flag("USE_CLEAN")            # fix broken characters in text
USE_STRUCT = flag("USE_STRUCT")          # section-aware + course-aware chunks
USE_PREFIX = flag("USE_PREFIX")          # nomic prefixes (search_document / search_query)
USE_TABLE = flag("USE_TABLE")            # read the course catalogue table
USE_OCR = flag("USE_OCR", "0")           # automatic OCR for pages without text (messy on this flowchart)
USE_FLOWCHART = flag("USE_FLOWCHART")    # hand-written text of image/table pages (flowchart_page10.txt, manual_page*.txt)
# Switches that change only retrieval / generation (no re-index needed)
USE_BM25 = flag("USE_BM25")              # hybrid search
USE_RERANK = flag("USE_RERANK")          # cross-encoder reranker
USE_STRICT = flag("USE_STRICT")          # strict prompt

STATE = {"rerank": USE_RERANK, "bm25": USE_BM25, "ce": None}

STRICT_PROMPT = """You answer questions about a university major handbook using ONLY the context below.

CONTEXT:
{context}
---
QUESTION: {question}

RULES:
1. Use only the context. Do not use outside knowledge.
2. If the context does not contain the answer, reply exactly: "I don't know based on the handbook."
3. Each excerpt has a page and section label. Do not mix different sections (for example Major and Minor requirements, or Core and Foundation courses).
4. If asked "how many", state the number found in the context, and list the items if the question asks for them.
5. Be concise: 1 to 4 sentences, or a short list. Mention the page number(s), like (page 7).

ANSWER:"""

BALANCED_PROMPT = """You answer questions about a university major handbook. Use the context below.

CONTEXT:
{context}
---
QUESTION: {question}

RULES:
1. Read ALL excerpts. If any excerpt contains the answer, give the answer from it, directly and completely (include every part the question asks for, for example both credits and noncredit items).
2. Only if NO excerpt mentions the topic, reply exactly: "I don't know based on the handbook."
3. Do not use outside knowledge and do not guess numbers.
4. Do not mix different sections (for example Major and Minor requirements, or Core and Foundation courses).
5. Be concise: 1 to 4 sentences, or a short list. Do not write page numbers.

ANSWER:"""

LOOSE_PROMPT = """You are a helpful AI assistant that answers questions based on provided documents.

CONTEXT FROM DOCUMENTS:
{context}
---
QUESTION: {question}

INSTRUCTIONS:
1. If the context contains relevant information, answer based on that.
2. If the context is insufficient, use your general knowledge.
3. Be concise - 2-4 sentences unless more detail is needed.

ANSWER:"""


# ---------------- TEXT CLEANING ----------------
def _description(m):
    return "Description" + ("s" if m.group(1) else "")


FIXES = [
    (r"Founda[;8'\u2019]onal", "Foundational"),
    (r"Educa[;8'\u2019]onal", "Educational"),
    (r"Objec[;8'\u2019]ves", "Objectives"),
    (r"Descrip[;8'\u2019]on(s?)", _description),
    (r"A[02]ributes", "Attributes"),
]


def clean(text: str) -> str:
    if not USE_CLEAN:
        return text
    for pat, rep in FIXES:
        text = re.sub(pat, rep, text)
    return re.sub(r"[ \t]+", " ", text)


# ---------------- SECTIONS AND COURSES ----------------
HEADINGS = [
    "The University Mission, Vision, and Graduate Attributes",
    "B.S. Computer Science: Program Description and Objectives",
    "Graduate Studies and Career Paths for the B.S. Computer Science",
    "Major Outline: Curricular Structure and Progression",
    "Sample Student Journeys for the Major and Minor",
    "Course Alignment with the Program Learning Outcomes",
    "Degree Requirements", "General Curriculum Requirements", "Major Requirements",
    "Minor Requirements", "Foundational Courses", "Intermediate Courses",
    "Advanced Courses", "CS 401/402: Capstone Project", "Program Learning Outcomes",
    "Program Educational Objectives", "Program Description",
    "Program Structure and Curriculum", "Teaching, Learning, and Assessment",
    "Course Catalogue", "Course Descriptions", "Mission", "Vision", "Graduate Attributes",
]
HEADING_RE = re.compile(
    r"(?:\d+\.\s*)?(?:" + "|".join(re.escape(h) for h in HEADINGS) + r")(?:\s*\([^)]*\))?\d*:?",
    re.IGNORECASE)
COURSE_TITLE_RE = re.compile(r"^(?:CS|MATH|ENG)\s?(?:\d{3}|3xx)(?:/\d{3})?:")


def is_heading(s: str) -> bool:
    return bool(s) and "...." not in s and bool(HEADING_RE.fullmatch(s))


def sectionize(docs: List[Document]) -> List[Document]:
    """Join all text pages, then cut at headings and at each course title.
    A course description that starts on one page and continues on the next stays in one piece."""
    out: List[Document] = []
    others: List[Document] = []
    lines = []  # (text, page, source)
    for d in docs:
        if d.metadata.get("kind", "text") == "text":
            for ln in d.page_content.splitlines():
                lines.append((ln, d.metadata["page"], d.metadata["source"]))
        else:  # table / OCR / flowchart documents carry their own label
            others.append(Document(page_content=d.page_content, metadata={**d.metadata}))

    base, sub, buf = "", "", []

    def emit():
        text = "\n".join(t for t, _p, _s in buf).strip()
        if text:
            label = base + (" > " + sub if sub else "")
            out.append(Document(page_content=text, metadata={
                "source": buf[0][2], "page": buf[0][1], "kind": "text", "section": label}))

    for ln, page, src in lines:
        s = ln.strip()
        if "...." in s:  # table of contents line: skip
            continue
        if is_heading(s):
            emit(); buf = []; base, sub = s.rstrip(":"), ""
        elif COURSE_TITLE_RE.match(s):
            emit(); buf = []; sub = s[:80]
        buf.append((ln, page, src))
    emit()
    return out + others


# ---------------- TABLE (course catalogue) ----------------
AVAIL = r"(Once every two(?:\s*years?)?|Once a year|Twice a year|On holding)"
CODE = r"^((?:CS|MATH|ENG)\s?\d{3}|CS3xx)\s+(.*)$"
TITLE_LINE = re.compile(r"^((?:CS|MATH|ENG)\s?(?:\d{3}|3xx)):\s*(.+)$")


def course_titles(texts: List[str]) -> dict:
    """Full course titles from the 'CS103: Programming for ...' headings."""
    titles = {}
    for t in texts:
        for line in t.splitlines():
            m = TITLE_LINE.match(line.strip())
            if m:
                titles[m.group(1).replace(" ", "")] = m.group(2).strip()
    return titles


def catalogue_docs(layout_text: str, page_no: int, name: str, titles: dict) -> List[Document]:
    """Turn the flattened catalogue table into one clear sentence per course."""
    rows, buf = [], ""

    def flush():
        nonlocal buf
        if not buf:
            return
        m = re.match(CODE, buf)
        if m:
            code, rest = m.groups()
            a = re.search(AVAIL, rest)
            if a:
                title = re.sub(r"\s+", " ", rest[:a.start()]).strip()
                full = titles.get(code.replace(" ", ""), "")
                if full and full.lower().startswith(title.lower()):
                    title = full  # the table wrapped a long title
                av = re.sub(r"\s+", " ", a.group(1))
                if av.lower().startswith("once every two"):
                    av = "Once every two years"
                prereq = re.sub(r"\s+", " ", rest[a.end():]).strip()
                prereq = re.sub(r"^years?\s*", "", prereq, flags=re.I)
                prereq = re.sub(r"(-level CS)$", r"\1 course", prereq) or "None"
                rows.append((code, title, av, prereq))
        buf = ""

    for line in layout_text.splitlines():
        s = re.sub(r"\s{2,}", "  ", line.strip())
        if not s:
            continue
        if re.match(CODE, s):
            flush()
            buf = s
        elif buf and not re.search(AVAIL, buf):
            buf += " " + s
        else:
            flush()
    flush()
    if not rows:
        return []

    meta = {"source": name, "page": page_no, "kind": "table", "section": "Course Catalogue table"}
    docs = [Document(
        page_content=code + " " + title + ": availability = " + av + "; prerequisites = " + pre,
        metadata=meta) for code, title, av, pre in rows]
    groups = {}
    for code, title, av, pre in rows:
        groups.setdefault(av, []).append(code + " " + title)
    for av, items in groups.items():
        docs.append(Document(
            page_content="Courses with availability '" + av + "' (" + str(len(items)) + "): " + "; ".join(items),
            metadata=meta))
    return docs


# ---------------- OCR (pages without text) ----------------
def ocr_page(pdf_path: str, idx: int) -> str:
    try:
        import numpy as np
        import pypdfium2 as pdfium
        img = pdfium.PdfDocument(pdf_path)[idx].render(scale=3).to_pil().convert("RGB")
        arr = np.array(img)
        try:
            from rapidocr_onnxruntime import RapidOCR
            res, _ = RapidOCR()(arr)
            items = [(b[0][1], b[0][0], t) for b, t, _s in (res or [])]
        except ImportError:
            from rapidocr import RapidOCR
            out = RapidOCR()(arr)
            items = [(b[0][1], b[0][0], t) for b, t in zip(out.boxes, out.txts)] if out.boxes is not None else []
    except Exception as e:
        print("  OCR skipped (" + str(e)[:80] + ")")
        return ""
    items.sort()
    lines, last_y = [], None
    for y, x, t in items:
        if last_y is None or abs(y - last_y) > 15:
            lines.append([])
        lines[-1].append((x, t))
        last_y = y
    return "\n".join(" ".join(t for _x, t in sorted(l)) for l in lines)


# ---------------- LOADING THE PDF ----------------
def load_pdf(pdf_path: str) -> List[Document]:
    reader = PdfReader(pdf_path)
    name = Path(pdf_path).name
    texts = [(p.extract_text() or "").strip() for p in reader.pages]
    titles = course_titles(texts)
    docs: List[Document] = []
    empty = []
    for i, (page, text) in enumerate(zip(reader.pages, texts)):
        if not text:
            empty.append(i)
            continue
        docs.append(Document(page_content=clean(text),
                             metadata={"source": name, "page": i + 1, "kind": "text"}))
        if USE_TABLE and re.search(AVAIL, text):
            layout = page.extract_text(extraction_mode="layout") or ""
            docs.extend(catalogue_docs(layout, i + 1, name, titles))
    if USE_OCR:
        for i in empty:
            text = ocr_page(pdf_path, i)
            if text.strip():
                print("  OCR page " + str(i + 1) + ": " + str(len(text)) + " characters")
                docs.append(Document(page_content=clean(text), metadata={
                    "source": name, "page": i + 1, "kind": "ocr",
                    "section": "Flowchart (text read from image by OCR)"}))
    if USE_FLOWCHART:
        fp = Path(__file__).with_name("flowchart_page10.txt")
        if fp.exists():
            docs.append(Document(page_content=fp.read_text(encoding="utf-8"), metadata={
                "source": name, "page": 10, "kind": "manual",
                "section": "Computer Science Flowchart (page 10, transcribed by hand)"}))
        else:
            print("  flowchart_page10.txt not found: skipped")
        # Tables that pypdf reads in the wrong order (pages 11 and 12): one file per table.
        # The page number is taken from the file name, e.g. manual_page11_plan.txt -> page 11.
        for mp in sorted(Path(__file__).parent.glob("manual_page*.txt")):
            m = re.match(r"manual_page(\d+)", mp.name)
            docs.append(Document(page_content=mp.read_text(encoding="utf-8"), metadata={
                "source": name, "page": int(m.group(1)), "kind": "manual",
                "section": "Hand-transcribed table (page " + m.group(1) + ")"}))

    print("  Loaded " + str(len(docs)) + " documents from " + name)
    return docs


# ---------------- EMBEDDINGS AND INDEX ----------------
class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text works better with task prefixes."""
    def embed_documents(self, texts):
        return super().embed_documents(["search_document: " + t for t in texts])

    def embed_query(self, text):
        return super().embed_query("search_query: " + text)


def make_embeddings():
    if USE_PREFIX and EMBED_MODEL.startswith("nomic"):
        return NomicEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)


def index_tag() -> str:
    raw = "|".join([EMBED_MODEL, str(CHUNK_SIZE), str(CHUNK_OVERLAP)] +
                   [str(int(x)) for x in (USE_CLEAN, USE_STRUCT, USE_PREFIX, USE_TABLE, USE_OCR, USE_FLOWCHART)])
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def load_and_index(pdf_path: str, embeddings, rebuild: bool = False) -> Chroma:
    tag = index_tag()
    store_dir = VECTOR_ROOT + "/" + Path(pdf_path).stem + "_v2_" + tag
    collection = "rag_v2_" + tag
    if Path(store_dir).exists() and not rebuild:
        print("Loading existing index: " + store_dir)
        return Chroma(collection_name=collection, embedding_function=embeddings,
                      persist_directory=store_dir)
    if Path(store_dir).exists():
        shutil.rmtree(store_dir)
    print("Indexing " + pdf_path + " ...")
    docs = load_pdf(pdf_path)
    if not docs:
        print("No text found in PDF.")
        sys.exit(1)
    if USE_STRUCT:
        docs = sectionize(docs)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)
    if USE_STRUCT:  # every chunk remembers which section it belongs to
        for c in chunks:
            sec = c.metadata.get("section", "")
            if sec:
                c.page_content = "[Section: " + sec + "]\n" + c.page_content
    print("  Indexing " + str(len(chunks)) + " chunks...")
    return Chroma.from_documents(documents=chunks, embedding=embeddings,
                                 collection_name=collection, persist_directory=store_dir)


# ---------------- RETRIEVAL ----------------
def tok(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Store:
    def __init__(self, chroma: Chroma):
        self.chroma = chroma
        data = chroma.get(include=["documents", "metadatas"])
        self.chunks = [Document(page_content=t, metadata=m or {})
                       for t, m in zip(data["documents"], data["metadatas"])]
        self.bm25 = None
        if STATE["bm25"]:
            try:
                from rank_bm25 import BM25Okapi
                self.bm25 = BM25Okapi([tok(c.page_content) for c in self.chunks])
            except ImportError:
                print("rank_bm25 not installed: BM25 off (pip install rank_bm25)")
                STATE["bm25"] = False


def rerank(question: str, docs: List[Document]):
    if STATE["ce"] is None:
        from sentence_transformers import CrossEncoder
        STATE["ce"] = CrossEncoder(RERANK_MODEL)
    scores = STATE["ce"].predict([(question, d.page_content) for d in docs])
    return sorted(zip(docs, [float(s) for s in scores]), key=lambda x: -x[1])


def retrieve(question: str, store: Store):
    fused, docs = {}, {}
    vec = store.chroma.similarity_search_with_score(question, k=CANDIDATES)
    for r, (d, _dist) in enumerate(vec):
        fused[d.page_content] = fused.get(d.page_content, 0) + 1.0 / (60 + r)
        docs[d.page_content] = d
    if store.bm25 is not None:
        scores = store.bm25.get_scores(tok(question))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:CANDIDATES]
        for r, i in enumerate(order):
            if scores[i] <= 0:
                break
            key = store.chunks[i].page_content
            fused[key] = fused.get(key, 0) + 1.0 / (60 + r)
            docs[key] = store.chunks[i]
    ranked = sorted(fused, key=lambda k: -fused[k])[:CANDIDATES]
    pairs = [(docs[k], fused[k]) for k in ranked]
    if STATE["rerank"]:
        try:
            pairs = rerank(question, [p[0] for p in pairs])
        except Exception as e:
            print("Reranker off (" + str(e)[:80] + ")")
            STATE["rerank"] = False
    return pairs[:TOP_K]


def ask(question: str, store: Store, llm: OllamaLLM) -> Dict[str, Any]:
    pairs = retrieve(question, store)
    sources = [{
        "source": str(d.metadata.get("source", "Unknown")),
        "page": d.metadata.get("page", "?"),
        "section": d.metadata.get("section", ""),
        "score": round(float(s), 4),
        "content": d.page_content,
    } for d, s in pairs]
    ordered = sorted(sources, key=lambda s: (s["page"] if isinstance(s["page"], int) else 0))
    context = "\n\n---\n\n".join(
        "[Page " + str(s["page"]) + "]\n" + s["content"] for s in ordered)
    if USE_STRICT:
        template = BALANCED_PROMPT if os.getenv("PROMPT") == "balanced" else STRICT_PROMPT
    else:
        template = LOOSE_PROMPT
    prompt = template.replace("{context}", context).replace("{question}", question)
    answer = llm.invoke(prompt).strip()
    return {"answer": answer, "sources": sources, "used_fallback": False}


def build_rag(pdf_path: str = DEFAULT_PDF, rebuild: bool = False):
    """Used by evaluate.py now, and by the Streamlit app later."""
    chroma = load_and_index(pdf_path, make_embeddings(), rebuild)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL,
                temperature=TEMPERATURE, num_ctx=NUM_CTX,
                num_predict=int(os.getenv("NUM_PREDICT", "350")),
                repeat_penalty=1.1,
                client_kwargs={"timeout": 180})
    return Store(chroma), llm


def config_summary() -> str:
    return ("llm=" + LLM_MODEL + " embed=" + EMBED_MODEL + " chunk=" + str(CHUNK_SIZE) +
            " top_k=" + str(TOP_K) + " clean=" + str(int(USE_CLEAN)) + " struct=" + str(int(USE_STRUCT)) +
            " prefix=" + str(int(USE_PREFIX)) + " table=" + str(int(USE_TABLE)) +
            " ocr=" + str(int(USE_OCR)) + " flow=" + str(int(USE_FLOWCHART)) +
            " bm25=" + str(int(USE_BM25)) + " rerank=" + str(int(USE_RERANK)) +
            " strict=" + str(int(USE_STRICT)) + " prompt=" + os.getenv("PROMPT", "strict"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", default=DEFAULT_PDF)
    p.add_argument("--question", "-q", default="What are the graduation requirements?")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--dump", action="store_true", help="write all chunks to chunks_dump.txt")
    a = p.parse_args()
    print("CONFIG: " + config_summary())
    store, llm = build_rag(a.pdf, a.rebuild)
    if a.dump:
        with open("chunks_dump.txt", "w", encoding="utf-8") as f:
            for i, c in enumerate(store.chunks):
                f.write("=== chunk " + str(i) + " | page " + str(c.metadata.get("page")) + " ===\n")
                f.write(c.page_content + "\n\n")
        print("Wrote chunks_dump.txt")
    print("Q: " + a.question)
    r = ask(a.question, store, llm)
    print("\nA: " + r["answer"] + "\n\nSources:")
    for s in r["sources"][:4]:
        print("   page " + str(s["page"]) + "  score=" + str(s["score"]) + "  " + s["section"])


if __name__ == "__main__":
    main()