"""Google provider for Allos Agent SDK.

This module provides integration with Google's Gemini and Vertex AI APIs.
It supports both the Gemini API and Vertex AI with multiple authentication methods:
- API key authentication for Gemini API
- Service account credentials for Vertex AI
- Application Default Credentials (ADC) for Vertex AI
- Service account impersonation for Vertex AI

Key classes:
- GoogleProvider: Main provider implementation with flexible auth support
"""

import sys

# Enforce Python 3.10+ requirement at import time
if sys.version_info < (3, 10):
    raise ImportError(
        f"Google provider requires Python 3.10 or higher.\n"
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}\n"
        f"Reason: google-auth dependency requires Python 3.10+\n"
        f"\nPlease either:\n"
        f"1. Upgrade to Python 3.10 or higher, OR\n"
        f"2. Use a different provider (openai, anthropic, ollama)"
    )

import json
import os
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from google.genai.pagers import Pager

from allos.providers.base import (
    BaseProvider,
    Message,
    MessageRole,
    ProviderChunk,
    ProviderResponse,
    ToolCall,
)
from allos.providers.metadata import MetadataBuilder
from allos.tools.base import BaseTool
from allos.utils.errors import ProviderError

from ..utils.logging import logger
from .registry import provider

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

# Only runs on Python 3.10
MODEL_CONTEXT_WINDOWS = {
    # Gemini 3.x models (Preview)
    "gemini-3-flash-preview": 1048576,
    "gemini-3-pro-preview": 1048576,
    # Gemini 2.5.x models
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    # Gemini 2.0.x models
    "gemini-2.0-flash": 1048576,
    # Gemini 1.5.x models
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-pro": 2097152,
    # Legacy
    "gemini-1.0-pro": 32768,
}


@provider("google")
class GoogleProvider(BaseProvider):
    """Google provider for Gemini API and Vertex AI integration.

    Supports flexible authentication including API keys for Gemini API,
    service account credentials, Application Default Credentials (ADC),
    and service account impersonation for Vertex AI.

    Attributes:
    ----------
    model : str
        The model identifier (e.g., "gemini-2.0-flash")
    vertexai : bool
        Whether to use Vertex AI instead of Gemini API
    project : Optional[str]
        GCP project ID for Vertex AI
    location : str
        GCP region for Vertex AI (default: "us-central1")
    client : genai.Client
        Initialized Google AI client

    Methods:
    -------
    chat(messages, tools)
        Send messages and receive a response with optional tool calling
    stream_chat(messages, tools)
        Stream a chat response with optional tool calling
    get_context_window()
        Get the model's context window size
    """

    env_var = "GOOGLE_API_KEY"

    @classmethod
    def check_env_config(cls) -> Tuple[bool, str]:
        """Check for Gemini API key or Vertex AI configuration."""
        # Priority 1: Check Gemini API Keys
        if os.environ.get("GOOGLE_API_KEY"):
            return (True, "GOOGLE_API_KEY (Set)")
        if os.environ.get("GEMINI_API_KEY"):
            return (True, "GEMINI_API_KEY (Set)")

        # Priority 2: Check Service Account JSON file
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if sa_path:
            if os.path.exists(sa_path):
                return (True, f"Service Account (File: {os.path.basename(sa_path)})")
            else:
                return (
                    False,
                    f"GOOGLE_APPLICATION_CREDENTIALS points to a non-existent file: {sa_path}",
                )

        # Priority 3: Check Vertex AI ADC configuration
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        if project:
            return (True, f"Vertex AI ADC (PROJECT={project}, LOCATION={location})")

        # Try to detect default ADC credentials
        try:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError

            creds, detected_project = google.auth.default()
            if detected_project:
                return (True, f"ADC (Auto-detected PROJECT={detected_project})")
        except DefaultCredentialsError:
            pass
        return (
            False,
            "No authentication method configured. See documentation for setup instructions.",
        )

    def __init__(
        self,
        model: str,
        sub_provider: str = "google",
        # Gemini API Auth (Backward compatible)
        api_key: Optional[str] = None,
        # Vertex AI mode flag (Backward compatible)
        vertexai: bool = False,
        # Vertex AI config
        project: Optional[str] = None,
        location: str = "us-central1",
        # Service Acc Auth options
        credentials: Optional["Credentials"] = None,
        credentials_path: Optional[str] = None,
        credentials_json: Optional[Union[Dict[str, Any], str]] = None,
        # Service Acc impersonation
        impersonate_service_account: Optional[str] = None,
        impersonation_scopes: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize Google provider with flexible authentication.

        Args:
            model: Model name (e.g., "gemini-2.0-flash").
            sub_provider: Name of sub provider (Vertex AI only).
            api_key: API key for Gemini API (not Vertex AI).
            vertexai: Use Vertex AI instead of Gemini API (Gemini API default).
            project: GCP project ID (auto-detected if not provided).
            location: GCP region for Vertex AI (default: us-central1).
            credentials: Pre-configured credentials object.
            credentials_path: Path to service account JSON file.
            credentials_json: Service account JSON as dict or string.
            impersonate_service_account: Service account email to impersonate.
            impersonation_scopes: OAuth scopes for impersonation.
            **kwargs: Additional keyword arguments passed to parent class.

        Examples:
            # Gemini API
            provider = GoogleProvider(model="gemini-2.0-flash", api_key="...")

            # Vertex AI with ADC
            provider = GoogleProvider(model="gemini-2.0-flash", vertexai=True)

            # Vertex AI with service account file
            provider = GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                credentials_path="/path/to/sa.json"
            )

            # Vertex AI with service account JSON
            provider = GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                credentials_json={"type": "service_account", ...}
            )

            # Vertex AI with impersonation
            provider = GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                impersonate_service_account="sa@project.iam.gserviceaccount.com"
            )
        """
        super().__init__(model, **kwargs)
        self._sub_provider = sub_provider
        self.vertexai = vertexai
        self.location = location
        self._project = project

        self._thought_signatures_used: bool = False

        # This will be populated by _verify_model_available()
        self._model_context_window: Optional[int] = None

        # Store auth parameters for credential loading
        self._api_key = api_key
        self._credentials = credentials
        self._credentials_path = credentials_path
        self._credentials_json = credentials_json
        self._impersonate_service_account = impersonate_service_account
        self._impersonation_scopes = impersonation_scopes or [
            "https://www.googleapis.com/auth/cloud-platform"
        ]

        try:
            if vertexai:
                # Load credentials and detect project
                creds, self.project = self._load_vertex_credentials()

                # Validate project is available
                if not self.project:
                    raise ProviderError(
                        "Vertex AI requires a project ID. Provide via:\n"
                        "  1. 'project' parameter\n"
                        "  2. GOOGLE_CLOUD_PROJECT environment variable\n"
                        "  3. 'project_id' in service account JSON\n"
                        "  4. Application Default Credentials",
                        provider="google",
                    )

                # Initialize Vertex AI client
                if creds:
                    self.client = genai.Client(
                        vertexai=True,
                        project=self.project,
                        location=location,
                        credentials=creds,
                    )
                else:
                    # Fall back to ADC
                    self.client = genai.Client(
                        vertexai=True, project=self.project, location=location
                    )
            else:
                # Gemini API mode
                self.project = None
                self.client = genai.Client(api_key=api_key)

            self._verify_model_available()
            auth_method = "Vertex AI" if self.vertexai else "Gemini API"
            logger.debug(f"Google {auth_method} provider initialized for '{model}'.")

        except ProviderError:
            raise

        except genai_errors.ClientError as e:
            # 4xx errors - likely auth/config issues
            raise ProviderError(
                f"Authentication or configuration error: {e.message}", provider="google"
            ) from e
        except genai_errors.ServerError as e:
            # 5xx errors - Google's problem
            raise ProviderError(
                f"Google API server error: {e.message}", provider="google"
            ) from e
        except genai_errors.APIError as e:
            # Catch-all for other API errors
            raise ProviderError(
                f"Google API error: {e.message}", provider="google"
            ) from e
        except Exception as e:
            # Non-API errors (network, etc.)
            raise ProviderError(
                f"Failed to initialize Google client: {e}", provider="google"
            ) from e

    def _load_vertex_credentials(self) -> Tuple[Optional[Any], Optional[str]]:
        """Load Vertex AI credentials following priority chain.

        Returns:
            Tuple of (credentials_object, project_id)
            - credentials_object: google.auth.credentials.Credentials or None for ADC
            - project_id: Detected project ID or None

        Priority:
            1. Explicit credentials object
            2. Service account JSON path (parameter)
            3. Service account JSON path (GOOGLE_APPLICATION_CREDENTIALS)
            4. Service account JSON content (parameter)
            5. Service account impersonation
            6. Application Default Credentials (ADC) - returns (None, project)
        """
        project = self._project or os.environ.get("GOOGLE_CLOUD_PROJECT")

        # Priority 1: Explicit credentials object
        if self._credentials:
            logger.debug("Using explicit credentials object")
            return self._credentials, project

        # Priority 2: Service account JSON path
        if self._credentials_path:
            logger.debug(f"Loading credentials from file: {self._credentials_path}")
            return self._load_credentials_from_file(self._credentials_path, project)

        # Priority 3: Service account JSON path (environment variable)
        env_sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if env_sa_path:
            logger.debug(
                f"Loading credentials from GOOGLE_APPLICATION_CREDENTIALS: {env_sa_path}"
            )
            return self._load_credentials_from_file(env_sa_path, project)

        # Priority 4: Service account JSON content
        if self._credentials_json:
            logger.debug("Loading credentials from JSON content")
            return self._load_credentials_from_json(self._credentials_json, project)

        # Priority 5: Service account impersonation
        if self._impersonate_service_account:
            logger.debug(
                f"Impersonating service account: {self._impersonate_service_account}"
            )
            return self._load_impersonated_credentials(project)

        # Priority 6: Application Default Credentials (ADC)
        logger.debug("Using Application Default Credentials (ADC)")
        return self._load_adc_credentials(project)

    def _load_credentials_from_file(
        self, file_path: str, project: Optional[str]
    ) -> Tuple[Any, Optional[str]]:
        """Load credentials from service account JSON file.

        Args:
            file_path: Path to service account JSON file
            project: Explicit project ID (or None to extract from file)

        Returns:
            Tuple of (credentials, project_id)
        """
        from google.oauth2 import service_account

        try:
            if not os.path.exists(file_path):
                raise ProviderError(
                    f"Service account file not found: {file_path}", provider="google"
                )

            creds = service_account.Credentials.from_service_account_file(
                file_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            # Extract project from JSON if not provided
            if not project:
                with open(file_path) as f:
                    sa_info = json.load(f)
                    project = sa_info.get("project_id")
                    if project:
                        logger.debug(
                            f"Extracted project from service account: {project}"
                        )
            return creds, project

        except json.JSONDecodeError as e:
            raise ProviderError(
                f"Invalid JSON in service account file {file_path}: {e}",
                provider="google",
            ) from e

        except Exception as e:
            raise ProviderError(
                f"Failed to load credentials from {file_path}: {e}", provider="google"
            ) from e

    def _load_credentials_from_json(
        self, credentials_json: Union[Dict[str, Any], str], project: Optional[str]
    ) -> Tuple[Any, Optional[str]]:
        """Load credentials from service account JSON content.

        Args:
            credentials_json: Service account JSON as dict or JSON string
            project: Explicit project ID (or None to extract from JSON)

        Returns:
            Tuple of (credentials, project_id)
        """
        from google.oauth2 import service_account

        try:
            # Parse JSON string to dict if needed
            if isinstance(credentials_json, str):
                sa_info = json.loads(credentials_json)
            else:
                sa_info = credentials_json

            # Validate required fields
            if not isinstance(sa_info, dict):
                raise ProviderError(
                    "credentials_json must be a dict or JSON string", provider="google"
                )

            creds = service_account.Credentials.from_service_account_info(
                sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            # Extract project from JSON if not provided
            if not project:
                project = sa_info.get("project_id")
                if project:
                    logger.debug(f"Extracted project from credentials JSON: {project}")

            return creds, project

        except json.JSONDecodeError as e:
            raise ProviderError(
                f"Invalid JSON in credentials_json: {e}", provider="google"
            ) from e
        except Exception as e:
            raise ProviderError(
                f"Failed to load credentials from JSON: {e}", provider="google"
            ) from e

    def _load_impersonated_credentials(
        self, project: Optional[str]
    ) -> Tuple[Any, Optional[str]]:
        """Load impersonated service account credentials.

        Args:
            project: Explicit project ID

        Returns:
            Tuple of (credentials, project_id)
        """
        import google.auth
        from google.auth import impersonated_credentials

        try:
            # Get source credentials (ADC or explicit)
            source_creds, detected_project = google.auth.default()

            # Use detected project if not explicitly provided
            if not project:
                project = detected_project

            # Create impersonated credentials
            creds = impersonated_credentials.Credentials(
                source_credentials=source_creds,
                target_principal=self._impersonate_service_account,
                target_scopes=self._impersonation_scopes,
            )

            logger.debug(
                f"Impersonating {self._impersonate_service_account} "
                f"with scopes: {self._impersonation_scopes}"
            )

            return creds, project

        except Exception as e:
            raise ProviderError(
                f"Failed to impersonate service account {self._impersonate_service_account}: {e}\n"
                f"Ensure source credentials have 'roles/iam.serviceAccountTokenCreator' role.",
                provider="google",
            ) from e

    def _load_adc_credentials(
        self, project: Optional[str]
    ) -> Tuple[None, Optional[str]]:
        """Use Application Default Credentials (ADC).

        Args:
            project: Explicit project ID

        Returns:
            Tuple of (None, project_id) - None signals ADC to genai.Client

        Raises:
            ProviderError: If ADC is not configured
        """
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        try:
            # Always verify ADC is available, even if project is provided
            _, detected_project = google.auth.default()

            # Use explicit project if provided, otherwise use detected project
            if not project:
                project = detected_project
                if project:
                    logger.debug(f"Auto-detected project from ADC: {project}")
            else:
                logger.debug(f"Using explicit project with ADC: {project}")

            return None, project

        except DefaultCredentialsError as e:
            # ADC is not configured - this is a critical error
            raise ProviderError(
                "Application Default Credentials (ADC) not found. "
                "To use Vertex AI, either:\n"
                "  1. Set up ADC: gcloud auth application-default login\n"
                "  2. Provide explicit credentials: credentials_path='/path/to/sa.json'\n"
                "  3. Set GOOGLE_APPLICATION_CREDENTIALS env var\n"
                "  4. Provide credentials as JSON: credentials_json={...}\n"
                f"Original error: {e}",
                provider="google",
            ) from e
        except Exception as e:
            logger.warning(f"Could not detect project from ADC: {e}")
            return None, project

    def _verify_model_available(self):
        """Check if the configured model is available.

        This method verifies that:
        1. The model name is valid and can be used with Gemini/Vertex AI APIs.
        2. Retrieves the model's actual context window size (if available)
        """
        model_name = "models/" + self.model
        if self.vertexai:
            model_name = "publishers/" + self._sub_provider + "/" + model_name
        try:
            # First check if the model is in the list of available models
            pulled_models = self.client.models.list()
            available_model_names = {m.name for m in pulled_models}
            if model_name not in available_model_names:
                # Find closest matching models for better error message
                suggestions = self._find_similar_models(self.model, pulled_models)

                error_msg = f"Model '{self.model}' not available."
                if suggestions:
                    error_msg += "\n\nDid you mean one of these?\n"
                    for suggestion in suggestions[:5]:  # Show top 5 suggestions
                        error_msg += f"  - {suggestion}\n"
                else:
                    shortened_models_list = list(pulled_models)
                    error_msg += (
                        f"\n\nAvailable models: {', '.join(sorted([self._extract_model_id(m.name) for m in shortened_models_list[:10] if m.name is not None]))}"
                        if pulled_models
                        else ""
                    )

                raise ProviderError(error_msg, provider="google")
            # Get detailed info
            model_info = next((m for m in pulled_models if m.name == model_name), None)
            # Extract context window from model info
            if model_info and model_info.input_token_limit:
                self._model_context_window = model_info.input_token_limit
                if self._model_context_window:
                    logger.debug(
                        f"Model '{self.model}' context window: "
                        f"{self._model_context_window} tokens"
                    )
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Could not verify model '{self.model}': {e.message}",
                provider="google",
            ) from e

    def _extract_model_id(self, full_model_name: str) -> str:
        """Extract clean model ID from full model name.

        Args:
            full_model_name: Full model name like 'models/gemini-2.0-flash' or
                           'publishers/google/models/gemini-2.0-flash'

        Returns:
            Clean model ID like 'gemini-2.0-flash'
        """
        if "/models/" in full_model_name:
            return full_model_name.split("/models/")[-1]
        return full_model_name

    def _find_similar_models(
        self, requested_model: str, available_models: Pager[types.Model]
    ) -> List[str]:
        """Find models similar to the requested model.

        Args:
            requested_model: The model name that was requested
            available_models: List of available model objects

        Returns:
            List of similar model names, sorted by similarity
        """
        import difflib

        # Extract clean model IDs from available models
        available_ids = [
            self._extract_model_id(m.name)
            for m in available_models
            if m.name is not None
        ]

        # Use difflib to find close matches
        # cutoff=0.4 means at least 40% similar
        close_matches = difflib.get_close_matches(
            requested_model, available_ids, n=5, cutoff=0.4
        )

        # If no close matches, suggest models with similar prefixes
        if not close_matches:
            requested_prefix = (
                requested_model.split("-")[0]
                if "-" in requested_model
                else requested_model
            )
            prefix_matches = [
                model_id
                for model_id in available_ids
                if model_id.startswith(requested_prefix)
            ]
            return sorted(prefix_matches)[:5]

        return close_matches

    def _convert_user_message(self, msg: Message) -> types.Content:
        """Convert a user message to Google format."""
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=msg.content or "")],
        )

    def _convert_assistant_message(self, msg: Message) -> types.Content:
        """Convert an assistant message to Google format."""
        parts: List[types.Part] = []

        if msg.content:
            parts.append(types.Part.from_text(text=msg.content))

        if msg.tool_calls:
            parts.extend(self._convert_tool_calls(msg))

        return types.Content(role="model", parts=parts)

    def _convert_tool_calls(self, msg: Message) -> List[types.Part]:
        """Convert tool calls to Google function call parts."""
        parts = []
        for tc in msg.tool_calls:
            fc_part = types.Part.from_function_call(name=tc.name, args=tc.arguments)

            # Attach thought signature if available
            if msg.thought_signatures and tc.id in msg.thought_signatures:
                thought_sig = msg.thought_signatures[tc.id]
                fc_part.thought_signature = thought_sig
                self._thought_signatures_used = True
                logger.debug(f"Including thought signature for {tc.id}")

            parts.append(fc_part)
        return parts

    def _convert_tool_message(self, msg: Message) -> types.Content:
        """Convert a tool result message to Google format."""
        return types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=msg.tool_call_id or "",
                    response={"result": msg.content},
                )
            ],
        )

    def _convert_messages(
        self,
        messages: List[Message],
    ) -> tuple[Optional[str], List[types.Content]]:
        """Convert Allos messages to google format."""
        system_instruction = None
        contents: List[types.Content] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = msg.content
            elif msg.role == MessageRole.USER:
                contents.append(self._convert_user_message(msg))
            elif msg.role == MessageRole.ASSISTANT:
                contents.append(self._convert_assistant_message(msg))
            elif msg.role == MessageRole.TOOL:
                contents.append(self._convert_tool_message(msg))
        return system_instruction, contents

    @staticmethod
    def _convert_tools(tools: List[BaseTool]) -> List[types.Tool]:
        """Convert Allos tools to Google FunctionDeclaration format.

        :param tools: Description
        :type tools: List[BaseTool]
        :return: Description
        :rtype: List[Tool]
        """
        function_declarations = []

        for tool in tools:
            properties: Dict[str, Dict[str, str]] = {}
            required = []

            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    required.append(param.name)
            func_decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            )
            function_declarations.append(func_decl)
        return [types.Tool(function_declarations=function_declarations)]

    def _parse_response(
        self,
        response: types.GenerateContentResponse,
    ) -> Tuple[Optional[str], List[ToolCall], Optional[Dict[str, bytes]]]:
        """Parse Google response to Allos format.

        Returns:
            Tuple of (content, tool_calls, thought_signatures)
        """
        content = None
        tool_calls = []
        thought_signatures = {}

        if response.candidates and len(response.candidates) > 0:
            if response.candidates[0].content and response.candidates[0].content.parts:
                # Extract text content and function calls from parts
                text_parts = []
                for idx, part in enumerate(response.candidates[0].content.parts):
                    # Extract text content
                    if part.text:
                        text_parts.append(part.text)

                    # Extract function calls
                    if part.function_call:
                        fc = part.function_call
                        tool_call_id = f"call_{fc.name}_{int(time.time() * 1000)}_{idx}"
                        tool_calls.append(
                            ToolCall(
                                id=tool_call_id,
                                name=fc.name or "unknown",
                                arguments=dict(fc.args) if fc.args else {},
                            )
                        )

                        if part.thought_signature:
                            thought_signatures[tool_call_id] = part.thought_signature
                            logger.debug(
                                f"Extracted thought signature for {tool_call_id}"
                            )

                # Combine text parts if any
                content = "".join(text_parts) if text_parts else None

        if thought_signatures:
            self._thought_signatures_used = True
            logger.debug("Thought signatures detected in response")

        return content, tool_calls, thought_signatures if thought_signatures else None

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send messages to the Google API and receive a response.

        Args:
            messages: List of messages in the conversation history.
            tools: Optional list of tools available for the model to call.
            **kwargs: Additional configuration options passed to GenerateContentConfig.

        Returns:
            ProviderResponse containing the model's response, tool calls,
            thought signatures (if used), and metadata.

        Raises:
            ProviderError: If the Google API request fails.
        """
        system_instruction, contents = self._convert_messages(messages)

        config_kwargs = {**kwargs}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)

        config = types.GenerateContentConfig(**config_kwargs)

        self._thought_signatures_used = False
        start_time = time.time()
        builder_kwargs = {
            "model": self.model,
            "contents": contents,
            "tools": tools or [],
        }
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )

            # Parse response first to extract thought signatures and set the flag
            content, tool_calls, thought_signatures = self._parse_response(response)

            # Build metadata after parsing so the flag is set correctly
            metadata = self._build_metadata(response, builder_kwargs, start_time)

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                thought_signatures=thought_signatures,
                metadata=metadata,
            )
        except genai_errors.ClientError as e:
            raise ProviderError(
                f"Google API client error ({e.code}): {e.message}", provider="google"
            ) from e
        except genai_errors.ServerError as e:
            raise ProviderError(
                f"Google API server error ({e.code}): {e.message}",
                provider="google",
            ) from e
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Google API error: {e}",
                provider="google",
            ) from e

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[ProviderChunk]:
        """Stream messages to the Google API and receive streamed responses.

        Args:
            messages: List of messages in the conversation history.
            tools: Optional list of tools available for the model to call.
            **kwargs: Additional configuration options passed to GenerateContentConfig.

        Yields:
            ProviderChunk objects containing streamed content, tool calls,
            thought signatures (if used), and final metadata.

        Raises:
            ProviderError: If the Google API streaming request fails.
        """
        config = self._prepare_stream_config(messages, tools, kwargs)
        builder_kwargs = self._build_request_metadata(messages, tools)
        self._thought_signatures_used = False
        start_time = time.time()

        try:
            accumulated_thought_signatures: Dict[str, bytes] = {}
            stream = self.client.models.generate_content_stream(
                model=self.model, contents=config["contents"], config=config["config"]
            )

            for chunk in stream:
                yield from self._process_stream_chunk(
                    chunk, accumulated_thought_signatures
                )

            # Yield final metadata chunk once at the end, after flag is set
            yield self._build_final_chunk(builder_kwargs, start_time)
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Google API streaming error: {e}",
                provider="google",
            ) from e

    def _prepare_stream_config(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare configuration for streaming request."""
        system_instruction, contents = self._convert_messages(messages)

        config_kwargs = {**kwargs}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)

        config = types.GenerateContentConfig(**config_kwargs)
        return {"contents": contents, "config": config}

    def _build_request_metadata(
        self, messages: List[Message], tools: Optional[List[BaseTool]]
    ) -> Dict[str, Any]:
        """Build metadata for request tracking."""
        _, contents = self._convert_messages(messages)
        return {
            "model": self.model,
            "contents": contents,
            "tools": tools or [],
        }

    def _process_stream_chunk(
        self,
        chunk: types.GenerateContentResponse,
        accumulated_thought_signatures: Dict[str, bytes],
    ) -> Iterator[ProviderChunk]:
        """Process a single streaming chunk."""
        # Extract text content from parts to avoid warning about non-text parts
        if chunk.candidates and chunk.candidates:
            candidate = chunk.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        yield ProviderChunk(content=part.text)

        # Process function calls
        if chunk.function_calls:
            yield from self._process_function_calls(
                chunk, accumulated_thought_signatures
            )

            # Yield accumulated thought signatures
            if accumulated_thought_signatures:
                self._thought_signatures_used = True
                yield ProviderChunk(thought_signatures=accumulated_thought_signatures)

    def _process_function_calls(
        self,
        chunk: types.GenerateContentResponse,
        accumulated_thought_signatures: Dict[str, bytes],
    ) -> Iterator[ProviderChunk]:
        """Extract and yield function calls from chunk."""
        if not (chunk.candidates and chunk.candidates):
            return

        candidate = chunk.candidates[0]
        if not (candidate.content and candidate.content.parts):
            return

        for idx, part in enumerate(candidate.content.parts):
            if not part.function_call:
                continue

            fc = part.function_call
            tool_call_id = f"call_{fc.name}_{int(time.time() * 1000)}_{idx}"

            # Store thought signature if present
            if part.thought_signature:
                accumulated_thought_signatures[tool_call_id] = part.thought_signature
                logger.debug(
                    f"Extracted thought signature for {tool_call_id} in streaming"
                )

            yield ProviderChunk(
                tool_call_done=ToolCall(
                    id=tool_call_id,
                    name=fc.name or "unknown",
                    arguments=dict(fc.args) if fc.args else {},
                )
            )

    def _build_metadata(self, response, builder_kwargs, start_time):
        """Build Metadata from Google response."""
        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        synthetic_response = {
            "id": "google_response",
            "model": self.model,
            "status": "completed",
            "usage": type(
                "Usage",
                (),
                {"input_tokens": input_tokens, "output_tokens": output_tokens},
            )(),
        }

        builder = MetadataBuilder(
            provider_name="google",
            request_kwargs=builder_kwargs,
            start_time=start_time,
        )

        google_specific = {
            "vertexai": self.vertexai,
            "project": self.project if self.vertexai else None,
            "location": self.location if self.vertexai else None,
            "used_thought_signatures": self._thought_signatures_used,
        }

        return (
            builder.with_response_obj(type("obj", (object,), synthetic_response)())
            .with_provider_specific(google=google_specific if google_specific else None)
            .build()
        )

    def _build_final_chunk(self, builder_kwargs, start_time):
        """Build final metadata chunk for streaming."""
        synthetic_response = {
            "id": "google_stream",
            "model": self.model,
            "status": "completed",
            "usage": type("Usage", (), {"input_tokens": 0, "output_tokens": 0})(),
        }

        builder = MetadataBuilder(
            provider_name="google",
            request_kwargs=builder_kwargs,
            start_time=start_time,
        )

        google_specific = {
            "vertexai": self.vertexai,
            "project": self.project if self.vertexai else None,
            "location": self.location if self.vertexai else None,
            "used_thought_signatures": self._thought_signatures_used,
        }

        metadata = (
            builder.with_response_obj(type("obj", (object,), synthetic_response)())
            .with_provider_specific(google=google_specific if google_specific else None)
            .build()
        )

        return ProviderChunk(final_metadata=metadata)

    def get_context_window(self) -> int:
        """Get the context window size for the configured model.

        Returns the model's maximum context window in tokens. If the model's
        context window was retrieved during initialization, returns that value.
        Otherwise, attempts to match the model against known context windows,
        falling back to a default of 4096 tokens for unknown models.

        Returns:
            int: The context window size in tokens.
        """
        if self._model_context_window:
            return self._model_context_window
        for model_prefix, size in MODEL_CONTEXT_WINDOWS.items():
            if model_prefix in self.model:
                return size
        return 4096  # Default for unknown models
