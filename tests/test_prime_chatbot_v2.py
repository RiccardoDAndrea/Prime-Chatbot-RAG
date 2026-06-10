import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from PrimeChatbotV2_LLM import PrimeChatbot, RAGAnswer, format_sources


class PrimeChatbotTests(unittest.TestCase):
    def make_chatbot(self, **overrides):
        options = {
            "file_path": ".",
            "persist_directory": tempfile.gettempdir(),
        }
        options.update(overrides)
        return PrimeChatbot(**options)

    def test_rejects_invalid_chunk_overlap(self):
        with self.assertRaisesRegex(ValueError, "chunk_overlap"):
            self.make_chatbot(chunk_size=100, chunk_overlap=100)

    def test_chunk_id_is_stable_and_source_sensitive(self):
        first = Document(
            page_content="same text",
            metadata={"source": "/a.pdf", "page": 1, "start_index": 0},
        )
        same = Document(page_content=first.page_content, metadata=dict(first.metadata))
        other_source = Document(
            page_content="same text",
            metadata={"source": "/b.pdf", "page": 1, "start_index": 0},
        )

        self.assertEqual(PrimeChatbot._chunk_id(first), PrimeChatbot._chunk_id(same))
        self.assertNotEqual(
            PrimeChatbot._chunk_id(first), PrimeChatbot._chunk_id(other_source)
        )

    def test_pdf_path_accepts_case_insensitive_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory, "document.PDF")
            pdf.touch()
            chatbot = self.make_chatbot(file_path=directory)
            self.assertEqual(chatbot.pdf_paths(), (pdf.resolve(),))

    def test_ask_returns_answer_and_deduplicated_sources(self):
        chatbot = self.make_chatbot()
        documents = [
            Document(
                page_content="ETL transforms before loading.",
                metadata={"file_name": "data.pdf", "page": 3},
            ),
            Document(
                page_content="ELT transforms after loading.",
                metadata={"file_name": "data.pdf", "page": 3},
            ),
        ]
        retriever = Mock()
        retriever.invoke.return_value = documents
        chain = Mock()
        chain.invoke.return_value = "  ETL and ELT differ. [1]  "
        chatbot.retriever = Mock(return_value=retriever)
        chatbot.ragchain = Mock(return_value=chain)

        result = chatbot.ask("What is the difference?")

        self.assertEqual(
            result,
            RAGAnswer(
                answer="ETL and ELT differ. [1]",
                sources=("data.pdf, page 3",),
            ),
        )
        prompt_input = chain.invoke.call_args.args[0]
        self.assertIn("[1] Source: data.pdf, page 3", prompt_input["documents"])

    def test_empty_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "question"):
            self.make_chatbot().ask("  ")

    def test_suggested_question_parser_handles_numbered_output(self):
        questions = PrimeChatbot._parse_suggested_questions(
            '1. "Was ist ETL?"\n2) Welche Vorteile hat ELT\n- Wie unterscheiden sie sich?',
            count=3,
        )

        self.assertEqual(
            questions,
            (
                "Was ist ETL?",
                "Welche Vorteile hat ELT?",
                "Wie unterscheiden sie sich?",
            ),
        )

    def test_suggested_questions_use_document_content(self):
        chatbot = self.make_chatbot()
        chatbot._doc_splits = [
            Document(
                page_content="MongoDB stores documents in collections.",
                metadata={"file_name": "mongo.pdf", "page": 1},
            )
        ]
        model = Mock()
        model.invoke.return_value = SimpleNamespace(
            content="1. Was ist MongoDB?\n2. Wie funktionieren Collections?"
        )
        chatbot.llm = Mock(return_value=model)

        questions = chatbot.suggested_questions(count=2)

        self.assertEqual(
            questions,
            ("Was ist MongoDB?", "Wie funktionieren Collections?"),
        )
        self.assertIn("MongoDB stores documents", model.invoke.call_args.args[0])

    def test_format_sources(self):
        self.assertEqual(format_sources(["a.pdf, page 1"]), "- a.pdf, page 1")


if __name__ == "__main__":
    unittest.main()
