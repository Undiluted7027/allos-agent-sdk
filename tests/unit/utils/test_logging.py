import logging
from pathlib import Path

from allos.utils.logging import THOUGHT_LEVEL, logger, setup_logging


class TestLoggingSetup:
    """Tests for the logging setup and custom logger."""

    def test_thought_logger_emits_record_when_enabled(self, configured_caplog):
        """
        Tests the ThoughtLogger's custom `thought` method.

        This test covers the `if self.isEnabledFor(THOUGHT_LEVEL):` branch by setting
        the log level low enough to capture THOUGHT-level messages.
        """
        # Set the log level to DEBUG (10), which is lower than THOUGHT (15)
        with configured_caplog.at_level(logging.DEBUG, logger="allos"):
            # Act: Log a message using the custom method
            logger.thought("This is an agent thought.")

            # Assert: Check that the log record was captured correctly
            assert len(configured_caplog.records) == 1
            record = configured_caplog.records[0]
            assert record.levelno == THOUGHT_LEVEL
            assert record.levelname == "THOUGHT"
            assert record.message == "This is an agent thought."

    def test_thought_logger_skips_record_when_disabled(self, configured_caplog):
        """
        Tests that the ThoughtLogger's `thought` method is a no-op when the
        log level is set too high. This implicitly tests the `if` condition is effective.
        """
        # Set the log level to INFO (20), which is higher than THOUGHT (15)
        with configured_caplog.at_level(logging.INFO, logger="allos"):
            # Act: Attempt to log a thought
            logger.thought("This thought should not be visible.")

            # Assert: No records should have been captured
            assert len(configured_caplog.records) == 0

    def test_setup_logging_with_file_handler(self, tmp_path: Path):
        """
        Tests that `setup_logging` correctly creates and writes to a log file.

        This test directly covers the `if log_file:` branch.
        """
        # Arrange: Define a path for the log file inside a temporary directory
        log_file_path = tmp_path / "test_run.log"

        # Act: Call setup_logging with the file path
        setup_logging(level="DEBUG", log_file=str(log_file_path))

        # Log some messages to be captured by the file handler
        logger.info("This is an info message for the file.")
        logger.warning("This is a warning.")

        # Assert: Check that the file was created and contains the logged messages
        assert log_file_path.exists()
        log_content = log_file_path.read_text()

        assert "INFO - This is an info message for the file." in log_content
        assert "WARNING - This is a warning." in log_content
