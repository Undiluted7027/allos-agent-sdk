# allos/context/__init__.py

"""Encapsulates the agent's conversational memory and state management.

This module provides the core data structures for managing an agent's short-term
memory. At its heart is the `ConversationContext` class, which maintains a
structured, chronological history of all interactions within a session, including
user prompts, assistant responses, tool call requests, and tool execution results.

This complete history is crucial for enabling the agent to conduct coherent,
multi-turn dialogues and to reason based on the outcomes of its previous actions.

Furthermore, the module's ability to serialize and deserialize the context is the
cornerstone of the SDK's session persistence feature, allowing agent
conversations to be saved to a file and resumed later.
"""

from .manager import ConversationContext

__all__ = ["ConversationContext"]
