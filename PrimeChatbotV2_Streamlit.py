"""Streamlit interface for PrimeChatbot V2."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import ollama
import streamlit as st

from PrimeChatbotV2_LLM import PrimeChatbot

APP_DATA = Path(".prime_chatbot")
UPLOAD_DIRECTORY = APP_DATA / "uploads"
DATABASE_DIRECTORY = APP_DATA / "chroma_db"
DEFAULT_PDF_DIRECTORY = Path("Prime_Chatbot_V1/PDF_docs")
DEFAULT_LLM = "qwen2.5:7b"
DEFAULT_EMBEDDING_MODEL = "granite-embedding:30m"
FALLBACK_QUESTIONS = (
    "Was sind die wichtigsten Aussagen der Dokumente?",
    "Fasse die zentralen Inhalte verständlich zusammen.",
    "Welche wichtigen Begriffe werden erklärt?",
    "Welche praktischen Beispiele enthalten die Dokumente?",
)
EMBEDDING_MODEL_MARKERS = (
    "embed",
    "embedding",
    "minilm",
    "bge-",
    "bge_",
    "e5-",
    "e5_",
    "nomic-embed",
)


def safe_file_name(file_name: str) -> str:
    """Return a filesystem-safe PDF name without directory components."""
    name = Path(file_name).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    return f"{stem or 'document'}.pdf"


def save_uploaded_pdfs(uploaded_files: Iterable) -> tuple[Path, tuple[str, ...]]:
    """Persist one upload set in an isolated, content-addressed directory."""
    uploads = []

    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()
        digest = hashlib.sha256(content).hexdigest()[:16]
        file_name = f"{digest}_{safe_file_name(uploaded_file.name)}"
        uploads.append((file_name, content))

    batch_identity = "\0".join(sorted(name for name, _ in uploads))
    batch_id = hashlib.sha256(batch_identity.encode("utf-8")).hexdigest()[:20]
    batch_directory = UPLOAD_DIRECTORY / batch_id
    batch_directory.mkdir(parents=True, exist_ok=True)

    saved_names = []
    for file_name, content in uploads:
        destination = batch_directory / file_name
        if not destination.exists() or destination.read_bytes() != content:
            destination.write_bytes(content)
        saved_names.append(file_name)

    return batch_directory, tuple(sorted(saved_names))


def collection_name(
    document_names: Iterable[str],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Build a stable Chroma collection name for one index configuration."""
    identity = "\0".join(
        (
            *sorted(document_names),
            embedding_model,
            str(chunk_size),
            str(chunk_overlap),
        )
    )
    return f"prime_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def classify_ollama_models(
    model_names: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split installed Ollama models into chat and embedding selections."""
    unique_names = sorted({name for name in model_names if name})
    embedding_models = tuple(
        name
        for name in unique_names
        if any(marker in name.lower() for marker in EMBEDDING_MODEL_MARKERS)
    )
    embedding_set = set(embedding_models)
    chat_models = tuple(name for name in unique_names if name not in embedding_set)
    return chat_models, embedding_models


@st.cache_data(ttl=30, show_spinner=False)
def installed_ollama_models() -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    """Read locally installed Ollama models with usable offline fallbacks."""
    try:
        response = ollama.list()
        names = tuple(model.model for model in response.models if model.model)
        chat_models, embedding_models = classify_ollama_models(names)
        if not chat_models:
            chat_models = (DEFAULT_LLM,)
        if not embedding_models:
            embedding_models = (DEFAULT_EMBEDDING_MODEL,)
        return chat_models, embedding_models, None
    except Exception as exc:
        return (
            (DEFAULT_LLM,),
            (DEFAULT_EMBEDDING_MODEL,),
            str(exc),
        )


def preferred_index(options: tuple[str, ...], preferred: str) -> int:
    """Return the preferred dropdown index when that model is installed."""
    return options.index(preferred) if preferred in options else 0


def complete_suggestions(
    generated: Iterable[str],
    count: int = 4,
) -> tuple[str, ...]:
    """Fill incomplete model output with useful general questions."""
    suggestions = []
    for question in (*generated, *FALLBACK_QUESTIONS):
        if question and question not in suggestions:
            suggestions.append(question)
        if len(suggestions) == count:
            break
    return tuple(suggestions)


def render_sources(sources: Iterable[str]) -> None:
    sources = tuple(sources)
    if not sources:
        return
    with st.expander("Quellen"):
        for source in sources:
            st.markdown(f"- {source}")


def reset_chat() -> None:
    st.session_state.messages = []


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chatbot", None)
    st.session_state.setdefault("index_signature", None)
    st.session_state.setdefault("indexed_files", ())
    st.session_state.setdefault("suggested_questions", ())


def main() -> None:
    st.set_page_config(
        page_title="PrimeChatbot V2",
        page_icon="P",
        layout="wide",
    )
    initialize_state()

    st.title("PrimeChatbot V2")
    st.caption("Lokaler PDF-Chat mit Ollama, ChromaDB und nachvollziehbaren Quellen.")

    with st.sidebar:
        st.header("Dokumente")
        source_mode = st.radio(
            "PDF-Quelle",
            ("PDFs hochladen", "Lokalen Ordner verwenden"),
        )

        uploaded_files = []
        local_path = str(DEFAULT_PDF_DIRECTORY)
        if source_mode == "PDFs hochladen":
            uploaded_files = st.file_uploader(
                "PDF-Dateien",
                type=("pdf",),
                accept_multiple_files=True,
                help="Die Dateien bleiben lokal auf diesem Rechner.",
            )
        else:
            local_path = st.text_input("PDF-Ordner oder PDF-Datei", local_path)

        st.header("Modelle")
        if st.button("Ollama-Modelle aktualisieren", use_container_width=True):
            installed_ollama_models.clear()
            st.rerun()

        chat_models, embedding_models, ollama_error = installed_ollama_models()
        if ollama_error:
            st.warning(
                "Ollama ist nicht erreichbar. Starte Ollama und aktualisiere "
                "anschließend die Modellliste."
            )

        model = st.selectbox(
            "LLM",
            chat_models,
            index=preferred_index(chat_models, DEFAULT_LLM),
            help="Lokal installierte Ollama-Modelle für die Antwortgenerierung.",
        )
        embedding_model = st.selectbox(
            "Embedding-Modell",
            embedding_models,
            index=preferred_index(embedding_models, DEFAULT_EMBEDDING_MODEL),
            help="Lokal installierte Modelle, die anhand ihres Namens als "
            "Embedding-Modelle erkannt wurden.",
        )

        with st.expander("Erweiterte Einstellungen"):
            chunk_size = st.slider("Chunk-Größe", 200, 2000, 800, 50)
            chunk_overlap = st.slider(
                "Chunk-Überlappung",
                0,
                min(500, chunk_size - 1),
                min(120, chunk_size - 1),
                10,
            )
            result_count = st.slider("Gefundene Textstellen", 1, 12, 6)
            temperature = st.slider(
                "Temperatur", 0.0, 1.0, 0.1, 0.05
            )

        index_clicked = st.button(
            "Dokumente indexieren",
            type="primary",
            use_container_width=True,
        )
        if st.button("Chat leeren", use_container_width=True):
            reset_chat()
            st.rerun()

    if index_clicked:
        try:
            if source_mode == "PDFs hochladen":
                if not uploaded_files:
                    raise ValueError("Bitte mindestens eine PDF-Datei hochladen.")
                pdf_path, document_names = save_uploaded_pdfs(uploaded_files)
                selected_paths = {
                    str((pdf_path / name).resolve()) for name in document_names
                }
            else:
                pdf_path = Path(local_path).expanduser()
                probe = PrimeChatbot(file_path=pdf_path)
                selected_paths = {str(path) for path in probe.pdf_paths()}
                document_names = tuple(
                    Path(path).name for path in sorted(selected_paths)
                )

            signature = collection_name(
                selected_paths,
                embedding_model,
                chunk_size,
                chunk_overlap,
            )
            chatbot = PrimeChatbot(
                file_path=pdf_path,
                model=model,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                k_int=result_count,
                temperature=temperature,
                persist_directory=DATABASE_DIRECTORY,
                collection_name=signature,
            )

            with st.spinner("PDFs werden gelesen und indexiert ..."):
                added = chatbot.add_only_new_docs_to_chroma()

            try:
                with st.spinner("Beispielfragen werden erstellt ..."):
                    suggestions = chatbot.suggested_questions(count=4)
            except Exception:
                suggestions = ()

            st.session_state.chatbot = chatbot
            st.session_state.index_signature = signature
            st.session_state.indexed_files = document_names
            st.session_state.suggested_questions = complete_suggestions(suggestions)
            reset_chat()
            st.success(
                f"{len(document_names)} PDF(s) bereit. "
                f"{added} neue oder geänderte Textabschnitte indexiert."
            )
        except Exception as exc:
            st.error(f"Indexierung fehlgeschlagen: {exc}")

    chatbot = st.session_state.chatbot
    if chatbot is None:
        st.info(
            "Wähle links PDFs aus und klicke auf "
            "**Dokumente indexieren**, um den Chat zu starten."
        )
    else:
        st.success(
            "Index aktiv: " + ", ".join(st.session_state.indexed_files),
            icon=None,
        )

    suggested_question = None
    if chatbot is not None and st.session_state.suggested_questions:
        st.subheader("Beispielfragen")
        st.caption("Klicke auf eine Frage, um sie direkt an die Dokumente zu stellen.")
        columns = st.columns(2)
        for index, suggestion in enumerate(st.session_state.suggested_questions):
            with columns[index % len(columns)]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{st.session_state.index_signature}_{index}",
                    use_container_width=True,
                ):
                    suggested_question = suggestion

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", ()))

    typed_question = st.chat_input(
        "Stelle eine Frage zu deinen PDFs",
        disabled=chatbot is None,
    )
    question = suggested_question or typed_question
    if question and chatbot is not None:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Antwort wird erstellt ..."):
                    result = chatbot.ask(question)
                st.markdown(result.answer)
                render_sources(result.sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                    }
                )
            except Exception as exc:
                st.error(
                    "Die Antwort konnte nicht erstellt werden. "
                    f"Prüfe, ob Ollama läuft und die Modelle installiert sind. ({exc})"
                )


if __name__ == "__main__":
    main()
