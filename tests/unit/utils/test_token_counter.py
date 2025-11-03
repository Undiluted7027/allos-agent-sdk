from unittest.mock import patch

import tiktoken

from allos.utils.token_counter import count_tokens, truncate_text_by_tokens


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

    def test_count_tokens_handles_unexpected_exception(self, configured_caplog):
        """
        Test that a generic exception from tiktoken is caught, logged,
        and triggers the fallback.
        """
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
