#!/usr/bin/env python3
"""
RAG System for PDF files.
Usage: python run_pdf.py --pdf major_handbook_pdfs/CS_MajorHandbook.pdf --question "Your question"
       python run_pdf.py --pdf <file.pdf> --rebuild    (index again)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# SETTINGS
DEFAULT_PDF = "major_handbook_pdfs/CS_MajorHandbook.pdf"
VECTOR_ROOT = "chroma_db"          # each PDF gets its own folder inside
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:0.5b")
TEMPERATURE = 0.2
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")  # Docker will override this
TOP_K = 5
SCORE_THRESHOLD = 0.3
SAMPLE_QUESTION = "What are the graduation requirements?"

PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on provided documents.

CONTEXT FROM DOCUMENTS:
{context}
---
QUESTION: {question}

INSTRUCTIONS:
1. If the context contains relevant information, answer based on that.
2. If the context is insufficient, use your general knowledge.
3. Be concise - 2-4 sentences unless more detail is needed.

ANSWER:"""


# TASK 1
def load_pdf(pdf_path: str) -> List[Document]:
    reader = PdfReader(pdf_path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:  # empty page, or a scanned image (needs OCR)
            continue
        documents.append(Document(
            page_content=text,
            metadata={"source": Path(pdf_path).name, "page": i + 1},
        ))
    print("  Loaded " + str(len(documents)) + " pages with text from " + Path(pdf_path).name)
    return documents


def vector_dir_for(pdf_path: str) -> str:
    return VECTOR_ROOT + "/" + Path(pdf_path).stem


def load_and_index(pdf_path: str, embeddings, rebuild: bool = False) -> Chroma:
    store_dir = vector_dir_for(pdf_path)
    collection = "rag_" + Path(pdf_path).stem.lower().replace(" ", "_")[:40]

    if Path(store_dir).exists() and not rebuild:
        print("Loading existing vector store: " + store_dir)
        return Chroma(collection_name=collection, embedding_function=embeddings,
                      persist_directory=store_dir)

    if rebuild and Path(store_dir).exists():
        import shutil
        shutil.rmtree(store_dir)

    print("Indexing " + pdf_path + " ...")
    documents = load_pdf(pdf_path)
    if not documents:
        print("No text found in PDF. It may be a scanned file (needs OCR).")
        sys.exit(1)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(documents)
    print("  Indexing " + str(len(chunks)) + " chunks...")
    return Chroma.from_documents(documents=chunks, embedding=embeddings,
                                 collection_name=collection, persist_directory=store_dir)


def ask(question: str, db: Chroma, llm: OllamaLLM) -> Dict[str, Any]:
    raw_results = db.similarity_search_with_score(question, k=TOP_K)
    sources = [
        {
            "source": str(doc.metadata.get("source", "Unknown")),
            "page": doc.metadata.get("page", "?"),
            "score": round(1 / (1 + score), 4),
            "content": doc.page_content,
        }
        for doc, score in raw_results
    ]
    avg_score = sum(s["score"] for s in sources) / len(sources) if sources else 0
    used_fallback = not sources or avg_score < SCORE_THRESHOLD

    if used_fallback:
        context = "No sufficiently relevant documents found. Use your general knowledge."
    else:
        context = "\n\n---\n\n".join(
            "[" + s["source"] + ", page " + str(s["page"]) + "]\n" + s["content"]
            for s in sources
        )
    prompt = PROMPT_TEMPLATE.replace("{context}", context).replace("{question}", question)
    answer = llm.invoke(prompt).strip()
    return {"answer": answer, "sources": sources, "used_fallback": used_fallback}


def build_rag(pdf_path: str = DEFAULT_PDF, rebuild: bool = False):
    """Used by run_pdf.py now, and by the Streamlit app later."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    db = load_and_index(pdf_path, embeddings, rebuild)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)
    return db, llm


def main():
    parser = argparse.ArgumentParser(description="Simple RAG Q&A (PDF files)")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the PDF file")
    parser.add_argument("--question", "-q", default=SAMPLE_QUESTION)
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the index")
    args = parser.parse_args()

    db, llm = build_rag(args.pdf, args.rebuild)
    print("Q: " + args.question)
    result = ask(args.question, db, llm)
    print("\nA: " + result["answer"] + "\n")
    if result["used_fallback"]:
        print("(answered from model knowledge - no relevant documents found)")
    else:
        print("Sources:")
        for s in result["sources"][:3]:
            print("   " + s["source"] + " (page " + str(s["page"]) + ")  score=" + str(s["score"]))


if __name__ == "__main__":
    main()