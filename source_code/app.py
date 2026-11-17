import csv
import os
import time
from pathlib import Path

os.environ.setdefault("PROMPT", "balanced")
os.environ.setdefault("LLM_MODEL", "qwen2.5:1.5b")

import streamlit as st
from langchain_ollama import OllamaLLM
import rag_v2
from rag_v2 import build_rag, ask, DEFAULT_PDF, OLLAMA_URL, TEMPERATURE, NUM_CTX

# model name -> text shown in the menu (accuracy numbers come from our 25-question evaluation)
MODELS = {
    "qwen2.5:1.5b": "1.5b - Recommended: fast and accurate (92%)",
    "qwen2.5:3b":   "3b - Most accurate (96%), but slower",
    "qwen2.5:0.5b": "0.5b - Fastest, but often wrong (not recommended)",
}
EXAMPLES = [
    "What is the major code of the B.S. Computer Science program?",
    "What is the prerequisite of CS311 Advanced Deep Learning?",
    "How many courses are in the Computer Science minor according to the flowchart?",
    "Which courses can I take after CS101?",
    "How much is the tuition fee for the Computer Science major?",
]
UPLOAD_DIR = Path("uploads")

st.set_page_config(page_title="Handbook Chatbot", page_icon="📘", layout="wide")


@st.cache_resource(show_spinner="Reading the PDF and building the index (first time only)...")
def load_store(pdf_path: str):
    store, _llm = build_rag(pdf_path, rebuild=False)
    return store


@st.cache_resource(show_spinner=False)
def load_llm(model_name: str):
    return OllamaLLM(
        model=model_name, base_url=OLLAMA_URL, temperature=TEMPERATURE, num_ctx=NUM_CTX,
        num_predict=int(os.getenv("NUM_PREDICT", "350")), repeat_penalty=1.1,
        client_kwargs={"timeout": 180},
    )


# ---------------- sidebar ----------------
with st.sidebar:
    page = st.radio("Page", ["Chat", "Evaluation results", "About"], label_visibility="collapsed")
    st.divider()
    st.header("Settings")
    model = st.selectbox("LLM model", list(MODELS), format_func=lambda m: MODELS[m],
                         help="A bigger model is more accurate but slower.")
    show_sources = st.checkbox("Show sources (page numbers)", value=True)

    st.subheader("Document")
    uploaded = st.file_uploader("Upload another PDF (optional)", type=["pdf"])
    pdf_path = DEFAULT_PDF
    if uploaded is not None:
        UPLOAD_DIR.mkdir(exist_ok=True)
        pdf_path = str(UPLOAD_DIR / uploaded.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded.getbuffer())
    st.caption("Using: " + Path(pdf_path).name)

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    with st.expander("Sources (" + str(len(sources)) + " passages)"):
        for s in sources:
            label = "Page " + str(s["page"])
            if s.get("section"):
                label += " | " + s["section"]
            st.markdown("**" + label + "**  (score " + str(s["score"]) + ")")
            st.text(s["content"][:500] + ("..." if len(s["content"]) > 500 else ""))


# ================= page: Chat =================
if page == "Chat":
    st.title("📘 Major Handbook Chatbot")
    st.write("Ask a question about the handbook. The answer comes only from the PDF.")

    if not st.session_state.messages:
        st.caption("Try one of these questions:")
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES):
            if col.button(ex, use_container_width=True):
                st.session_state.pending = ex
                st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"].replace("\n", "  \n"))  # keep line breaks
            if m["role"] == "assistant":
                if m.get("meta"):
                    st.caption(m["meta"])
                if show_sources and m.get("sources"):
                    render_sources(m["sources"])

    typed = st.chat_input("Type your question here, e.g. What is the prerequisite of CS311?")
    question = typed or st.session_state.pop("pending", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                store = load_store(pdf_path)
                llm = load_llm(model)
                with st.spinner("Thinking..."):
                    t0 = time.time()
                    result = ask(question, store, llm)
                    secs = round(time.time() - t0, 1)
                answer, sources = result["answer"], result["sources"]
                meta = "Model: " + model + " | " + str(secs) + " s"
                st.markdown(answer.replace("\n", "  \n"))  # keep line breaks
                st.caption(meta)
                if show_sources:
                    render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta, "sources": sources})
            except Exception as e:
                msg = ("Sorry, something went wrong: " + str(e)[:300] +
                       "\n\nCheck that Ollama is running and the model is pulled "
                       "(ollama pull " + model + ").")
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

# ================= page: Evaluation results =================
elif page == "Evaluation results":
    st.title("Evaluation results")
    st.write("25 test questions about the handbook. Each answer was labeled "
             "correct, incomplete, ambiguous or incorrect.")
    f = Path("comparison.csv")
    if f.exists():
        rows = list(csv.reader(open(f, encoding="utf-8-sig")))
        block = []
        for r in rows:
            if not r or not any(r):
                break
            block.append(r)
        st.dataframe([dict(zip(block[0], r)) for r in block[1:]], use_container_width=True)
        st.caption("Run A-B: baseline. C-E: improved pipeline (v2). F-G: improved pipeline + balanced prompt (v3). "
                   "H-L: v3 with one part switched off (ablation).")
    else:
        st.info("comparison.csv was not found. Run `python compare.py` first.")

# ================= page: About =================
else:
    st.title("About")
    st.markdown(
        "- **Documents:** the Computer Science major handbook (PDF).\n"
        "- **How it works:** the question is searched in the PDF (vector search + keyword search, "
        "then re-ranked). The best passages are sent to a local LLM (Ollama), which writes the answer.\n"
        "- **Everything runs locally.** No data leaves your computer.\n"
        "- **Each question is answered on its own.** The chat history is for you to read; "
        "it is not sent to the model.\n\n"
        "Current settings: `" + rag_v2.config_summary() + "`"
    )