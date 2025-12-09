# tests/unit/providers/test_metadata.py

from unittest.mock import MagicMock

import pytest

from allos.providers.metadata import EstimatedCost, MetadataBuilder


@pytest.fixture
def mock_response():
    """Provides a basic mock response object."""
    response = MagicMock()
    response.id = "resp_123"
    response.model = "gpt-4"
    response.status = "completed"
    response.usage.input_tokens = 10
    response.usage.output_tokens = 20
    response.usage.input_tokens_details = None
    response.usage.prompt_tokens_details = None
    return response


class TestMetadataBuilder:
    def test_build_raises_error_if_no_response_obj(self):
        """Test the 'if not self._response_obj:' branch in build()."""
        builder = MetadataBuilder(provider_name="test", request_kwargs={}, start_time=0)
        with pytest.raises(
            ValueError, match="A response object must be provided to build metadata."
        ):
            builder.build()

    def test_build_model_info_parses_version_string(self):
        """Test the try block of _build_model_info for version parsing."""
        builder = MetadataBuilder(provider_name="test", request_kwargs={}, start_time=0)
        # Call the private method directly for this unit test
        model_info = builder._build_model_info(model_id="gpt-4o-2024-08-06")

        assert model_info.model_id == "gpt-4o-2024-08-06"
        assert model_info.model_version == "2024-08-06"

    def test_build_usage_info_with_prompt_completion_tokens(self):
        """Test the 'elif' branch of _build_usage_info for cache tokens."""
        builder = MetadataBuilder(provider_name="test", request_kwargs={}, start_time=0)

        # Mock a response object that uses the Chat Completions API format
        usage_obj = MagicMock()

        # Explicitly set the attributes that will be checked first to None
        usage_obj.input_tokens = None
        usage_obj.output_tokens = None
        usage_obj.input_tokens_details = None

        # Set the attributes for the fallback logic
        usage_obj.prompt_tokens = 30
        usage_obj.completion_tokens = 40

        # Configure the nested attribute for the 'elif' branch
        usage_obj.prompt_tokens_details = MagicMock()
        usage_obj.prompt_tokens_details.cached_tokens = 5

        # Call the private method directly
        usage_info = builder._build_usage_info(model_id="gpt-4", usage_obj=usage_obj)

        assert usage_info.input_tokens == 30
        assert usage_info.output_tokens == 40
        assert usage_info.cache_usage
        assert usage_info.cache_usage.cache_read_tokens == 5
        assert usage_info.cache_usage.cache_hit_rate == 5 / 30

    def test_calculate_cost_returns_none_for_unknown_model(self):
        """Test the 'if not model_key:' branch in _calculate_cost."""
        # Provider 'openai' is known, but model 'unknown-model' is not in the pricing table
        builder = MetadataBuilder(
            provider_name="openai", request_kwargs={}, start_time=0
        )

        cost = builder._calculate_cost(
            model_id="unknown-model-xyz", input_tokens=100, output_tokens=100
        )

        assert cost is None

    def test_build_quality_signals_detects_refusal(self, mock_response):
        """Test the 'refusal = True' branch of _build_quality_signals."""
        # Set up a mock response with a 'refusal' content part
        refusal_part = MagicMock(type="refusal")
        message_item = MagicMock(type="message", content=[refusal_part])
        mock_response.output = [message_item]

        builder = MetadataBuilder(provider_name="test", request_kwargs={}, start_time=0)
        builder._response_obj = (
            mock_response  # Set the response object for the method to use
        )

        quality_signals = builder._build_quality_signals(status="completed")

        assert quality_signals.refusal_detected is True

    def test_build_quality_signals_handles_incomplete_status(self, mock_response):
        """Test the 'if status_str == "incomplete"' branch of _build_quality_signals."""
        # Set up a mock response with 'incomplete_details'
        mock_response.incomplete_details = MagicMock(reason="max_tokens_hit")

        builder = MetadataBuilder(provider_name="test", request_kwargs={}, start_time=0)
        builder._response_obj = mock_response

        # Pass "incomplete" as the status
        quality_signals = builder._build_quality_signals(status="incomplete")

        assert quality_signals.finish_reason == "max_tokens_hit"

    def test_calculate_cost_success_path(self):
        """Test the successful cost calculation path in _calculate_cost."""
        # Use a known provider and model from the _STATIC_PRICING table
        builder = MetadataBuilder(
            provider_name="openai", request_kwargs={}, start_time=0
        )

        # The model_id 'gpt-4o-latest' contains the key 'gpt-4o'
        cost = builder._calculate_cost(
            model_id="gpt-4o", input_tokens=100_000, output_tokens=200_000
        )

        assert isinstance(cost, EstimatedCost)
        # Expected input cost: (100,000 / 1,000,000) * $5.00 = $0.50
        assert cost.input_cost_usd == pytest.approx(0.5)
        # Expected output cost: (200,000 / 1,000,000) * $15.00 = $3.00
        assert cost.output_cost_usd == pytest.approx(3.0)
        # Expected total cost
        assert cost.total_usd == pytest.approx(3.5)
