import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import _available_flash_fallbacks, _invoke_with_model_fallback
from langchain_google_genai.chat_models import GoogleModelNotFoundError


class GeminiModelFallbackTests(unittest.TestCase):
    def test_available_flash_fallbacks_are_newest_first(self) -> None:
        models = [
            SimpleNamespace(
                name="models/gemini-2.0-flash",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-2.5-flash",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-3.8-flash-preview-01-15",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-3.8-flash",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-4.0-flash-lite",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-4.0-pro",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-4.0-flash",
                supported_actions=["countTokens"],
            ),
        ]

        self.assertEqual(
            _available_flash_fallbacks(models),
            [
                "gemini-3.8-flash",
                "gemini-3.8-flash-preview-01-15",
                "gemini-2.5-flash",
            ],
        )

    @patch("google.genai.Client")
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_unavailable_model_falls_back_to_newest_available_flash(
        self,
        chat_model: MagicMock,
        genai_client: MagicMock,
    ) -> None:
        genai_client.return_value.models.list.return_value = [
            SimpleNamespace(
                name="models/gemini-3.8-flash",
                supported_actions=["generateContent"],
            )
        ]

        def create_model(*, model: str, **_: object) -> MagicMock:
            instance = MagicMock()
            if model == "gemini-2.5-flash":
                instance.invoke.side_effect = GoogleModelNotFoundError("not found")
            else:
                instance.invoke.return_value = "answer"
            return instance

        chat_model.side_effect = create_model

        result = _invoke_with_model_fallback("prompt", "api-key", "gemini-2.5-flash")

        self.assertEqual(result, "answer")
        self.assertEqual(
            [call.kwargs["model"] for call in chat_model.call_args_list],
            ["gemini-2.5-flash", "gemini-3.8-flash"],
        )


if __name__ == "__main__":
    unittest.main()
