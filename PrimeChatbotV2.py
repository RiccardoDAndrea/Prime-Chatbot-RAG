"""Command-line entry point for PrimeChatbot V2."""

from __future__ import annotations

import argparse
import logging

from PrimeChatbotV2_LLM import PrimeChatbot, format_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index local PDFs and ask a grounded question with Ollama."
    )
    parser.add_argument("question", help="Question to answer from the PDFs")
    parser.add_argument(
        "--pdf-path",
        default="Prime_Chatbot_V1/PDF_docs",
        help="PDF file or directory containing PDFs",
    )
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--embedding-model", default="granite-embedding:30m")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--results", type=int, default=6)
    parser.add_argument("--database", default="chroma_db")
    parser.add_argument("--collection", default="prime_chatbot_v2")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Use the existing index without scanning PDFs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    chatbot = PrimeChatbot(
        file_path=args.pdf_path,
        model=args.model,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k_int=args.results,
        persist_directory=args.database,
        collection_name=args.collection,
    )

    if not args.skip_index:
        added = chatbot.add_only_new_docs_to_chroma()
        print(f"Indexed {added} new or changed chunk(s).")

    result = chatbot.ask(args.question)
    print(f"\n{result.answer}")
    if result.sources:
        print(f"\nSources:\n{format_sources(result.sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
