import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from PrimeChatbotV2_Streamlit import (
    classify_ollama_models,
    collection_name,
    complete_suggestions,
    preferred_index,
    safe_file_name,
    save_uploaded_pdfs,
)


class StreamlitHelperTests(unittest.TestCase):
    def test_safe_file_name_removes_directories_and_special_characters(self):
        self.assertEqual(safe_file_name("../../My report (1).pdf"), "My_report_1.pdf")

    def test_collection_name_is_stable_and_configuration_sensitive(self):
        first = collection_name(("b.pdf", "a.pdf"), "embed", 800, 120)
        reordered = collection_name(("a.pdf", "b.pdf"), "embed", 800, 120)
        changed = collection_name(("a.pdf", "b.pdf"), "embed", 900, 120)

        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)

    def test_ollama_models_are_split_into_chat_and_embedding_models(self):
        chat_models, embedding_models = classify_ollama_models(
            (
                "qwen2.5:7b",
                "granite-embedding:30m",
                "all-minilm:latest",
                "mxbai-embed-large:latest",
            )
        )

        self.assertEqual(chat_models, ("qwen2.5:7b",))
        self.assertEqual(
            embedding_models,
            (
                "all-minilm:latest",
                "granite-embedding:30m",
                "mxbai-embed-large:latest",
            ),
        )

    def test_preferred_model_is_selected_when_available(self):
        self.assertEqual(preferred_index(("a", "preferred"), "preferred"), 1)
        self.assertEqual(preferred_index(("a", "b"), "missing"), 0)

    def test_incomplete_suggestions_are_filled(self):
        suggestions = complete_suggestions(("Was ist MongoDB?",))

        self.assertEqual(len(suggestions), 4)
        self.assertEqual(suggestions[0], "Was ist MongoDB?")

    def test_uploaded_sets_are_saved_in_isolated_directories(self):
        with TemporaryDirectory() as directory:
            with patch(
                "PrimeChatbotV2_Streamlit.UPLOAD_DIRECTORY", Path(directory)
            ):
                first = SimpleNamespace(name="one.pdf", getvalue=lambda: b"one")
                second = SimpleNamespace(name="two.pdf", getvalue=lambda: b"two")

                first_directory, _ = save_uploaded_pdfs([first])
                second_directory, _ = save_uploaded_pdfs([second])

                self.assertNotEqual(first_directory, second_directory)
                self.assertEqual(len(tuple(first_directory.glob("*.pdf"))), 1)
                self.assertEqual(len(tuple(second_directory.glob("*.pdf"))), 1)

    def test_app_starts_without_indexing_or_ollama(self):
        app = AppTest.from_file("PrimeChatbotV2_Streamlit.py")
        app.run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "PrimeChatbot V2")
        self.assertTrue(app.chat_input[0].disabled)


if __name__ == "__main__":
    unittest.main()
