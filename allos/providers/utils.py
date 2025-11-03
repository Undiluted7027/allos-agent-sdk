# allos/providers/utils.py
"""
Utility functions that are common across LLM providers.
"""


def _init_metadata(total_items: int) -> dict[str, dict[str, int]]:
    return {
        "messages": {"total": 0, "processed": 0, "skipped": 0},
        "tool_calls": {"total": 0, "processed": 0, "skipped": 0},
        "overall": {"total": total_items, "processed": 0, "skipped": 0},
    }
