from unittest.mock import patch

import tiktoken

from allos.utils.token_counter import (
    _get_encoding_for_ollama_model,
    count_tokens,
    truncate_text_by_tokens,
)


class TestCountTokens:
    """Tests for the count_tokens utility."""

    def test_count_tokens_with_known_model(self):
        """Test the token counting utility with a model known by tiktoken."""
        text = "hello world"
        # gpt-4 tokenizer sees "hello" and " world" as two separate tokens
        assert count_tokens(text, model="gpt-4") == 2

    def test_count_tokens_with_fallback_model(self):
        """Test that an unknown model name triggers the character-based fallback."""
        text = "This is a sentence with eight words."  # 36 chars
        # Fallback is len(text) // 4 = 36 // 4 = 9
        assert count_tokens(text, model="a-fake-model-name") == 9

    def test_count_tokens_with_empty_string(self):
        """Test that an empty string correctly returns 0 tokens."""
        assert count_tokens("", model="gpt-4") == 0
        assert count_tokens("", model="a-fake-model-name") == 0

    def test_count_tokens_with_ollama_model_exact_match(self, configured_caplog):
        """Test counting tokens with an Ollama model that has an exact encoding match."""
        # Clear cache to ensure fresh execution
        from allos.utils.token_counter import _get_encoding
        _get_encoding.cache_clear()

        text = "hello world"
        # llama3.1:8b should map to cl100k_base encoding
        result = count_tokens(text, model="llama3.1:8b")
        assert result > 0  # Should successfully count tokens
        assert "Using 'cl100k_base' encoding for Ollama model" in configured_caplog.text

    def test_count_tokens_with_ollama_model_prefix_match(self, configured_caplog):
        """Test counting tokens with an Ollama model using prefix matching."""
        # Clear cache to ensure fresh execution
        from allos.utils.token_counter import _get_encoding
        _get_encoding.cache_clear()

        text = "hello world"
        # mistral-custom should match 'mistral' prefix
        result = count_tokens(text, model="mistral:custom-version")
        assert result > 0
        assert "Using 'cl100k_base' encoding for Ollama model" in configured_caplog.text

    def test_count_tokens_handles_unexpected_exception(self, configured_caplog):
        """
        Test that a generic exception from tiktoken is caught, logged,
        and triggers the fallback.
        """
        # Clear the LRU cache to ensure the exception is not cached
        from allos.utils.token_counter import _get_encoding
        _get_encoding.cache_clear()

        text = "some text to count"  # 18 chars
        with patch(
            "allos.utils.token_counter.tiktoken.encoding_for_model",
            side_effect=Exception("Simulated tiktoken error"),
        ):
            # The function should fall back to char-based counting: 18 // 4 = 4
            result = count_tokens(text, model="gpt-4")
            assert result == 4

        # Assert that the warning was logged
        assert "An unexpected error occurred with tiktoken" in configured_caplog.text
        assert "Simulated tiktoken error" in configured_caplog.text
        assert configured_caplog.records[0].levelname == "WARNING"


class TestOllamaModelEncodingMapping:
    """Tests for Ollama model to tiktoken encoding mapping."""

    def test_get_encoding_for_ollama_model_exact_match(self):
        """Test exact match for Ollama model names."""
        # Test exact matches
        assert _get_encoding_for_ollama_model("llama3.1") == "cl100k_base"
        assert _get_encoding_for_ollama_model("mistral") == "cl100k_base"
        assert _get_encoding_for_ollama_model("qwen2") == "cl100k_base"

    def test_get_encoding_for_ollama_model_with_version_tag(self):
        """Test that version tags are stripped correctly."""
        assert _get_encoding_for_ollama_model("llama3.1:8b") == "cl100k_base"
        assert _get_encoding_for_ollama_model("mistral:latest") == "cl100k_base"

    def test_get_encoding_for_ollama_model_prefix_match(self):
        """Test prefix matching for custom model variants."""
        assert _get_encoding_for_ollama_model("llama3.1-custom") == "cl100k_base"
        assert _get_encoding_for_ollama_model("mistral-finetune") == "cl100k_base"

    def test_get_encoding_for_ollama_model_no_match(self):
        """Test that unknown models return None."""
        assert _get_encoding_for_ollama_model("totally-unknown-model") is None


class TestTruncateTextByTokens:
    """Tests for the truncate_text_by_tokens utility."""

    def test_truncate_not_needed(self):
        """Test that text is returned unmodified if it's within the token limit."""
        text = "hello world"  # 2 tokens
        result = truncate_text_by_tokens(text, max_tokens=10, model="gpt-4")
        assert result == text

    def test_truncate_with_known_model(self):
        """Test truncation using a known tiktoken model."""
        text = "this is a long sentence for testing truncation"  # 8 tokens
        # We expect it to truncate to the first 4 tokens: "this is a long"
        result = truncate_text_by_tokens(text, max_tokens=4, model="gpt-4")
        assert result == "this is a long"
        assert count_tokens(result, model="gpt-4") <= 4

    def test_truncate_with_fallback_model(self):
        """Test truncation using the character-based fallback."""
        text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
        # Fallback logic: max_chars = 4 * 4 = 16
        result = truncate_text_by_tokens(text, max_tokens=4, model="a-fake-model-name")
        assert result == "abcdefghijklmnop"
        assert len(result) == 16

    def test_truncate_with_unicode_characters(self):
        """
        Test that truncation handles multi-byte characters correctly.
        The tiktoken library is designed to handle this safely.
        """
        text = "Hello 👋, this is a test with an emoji 😊."  # 12 tokens
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(text)
        truncated = encoding.decode(tokens[:7])
        assert truncate_text_by_tokens(text, 7, "gpt-4") == truncated

    def test_truncate_handles_exception_fallback(self):
        """Test that exceptions in truncation trigger character-based fallback."""
        text = "test text for truncation"  # 24 chars
        with patch(
            "allos.utils.token_counter._get_encoding",
            side_effect=Exception("Simulated error"),
        ):
            # Should fall back to character-based: 4 tokens * 4 chars = 16 chars
            result = truncate_text_by_tokens(text, max_tokens=4, model="gpt-4")
            assert result == "test text for tr"
            assert len(result) == 16
