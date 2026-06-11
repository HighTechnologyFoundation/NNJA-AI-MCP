import json
import logging
import os
import shlex
from typing import Any

import mcp.types as types
from dotenv import load_dotenv
from fuzzywuzzy import process
from google import genai
from mcp import ClientSession
from pydantic import AnyUrl

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


class GeminiQueryHandler:
    """Handle Gemini API interaction and MCP tool execution."""

    def __init__(self, client_session: ClientSession, model: str | None = None):
        self.client_session = client_session
        if not (api_key := os.getenv("GEMINI_API_KEY")):
            raise RuntimeError(
                "GEMINI_API_KEY is not set, so the AI chat client can't start.\n"
                "\n"
                "To fix this:\n"
                "    1. Copy .env.template to .env\n"
                "    2. Add GEMINI_API_KEY=<your key> "
                "(get one at https://aistudio.google.com/apikey)"
                "\n"
                "Don't have a key? You can still explore the server without one:\n"
                "   uv run mcp-client --members"
            )

        # Initialize the Gemini client
        self.gemini = genai.Client(api_key=api_key)

        # Use the model provided by CLI or env var or the default
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL

        # Create an asynchronous chat session with MCP tools enabled for the model
        self.chat = self.gemini.aio.chats.create(
            model=self.model,
            config=genai.types.GenerateContentConfig(
                tools=[self.client_session],  # Expose MCP tools to the LLM
            ),
        )

        self._prompt_listing: list[types.Prompt] | None = None
        self._resource_listing: list[types.Resource] | None = None
        self._resource_contents: dict[str, Any] = {}

    async def verify_model(self) -> None:
        """Fail fast if the configured model can't server generateContent."""
        try:
            available = [
                m.name.removeprefix("models/")
                async for m in await self.gemini.aio.models.list()
                if "generateContent" in (m.supported_actions or [])
                and m.name is not None
            ]
        except Exception as e:
            logger.warning("Could not verify model %s: %s", self.model, e)
            return

        if self.model.removeprefix("models/") not in available:
            suggestions = process.extract(self.model, available, limit=10)
            preview = "\n".join(f"  - {name}" for name, _score in suggestions)
            raise RuntimeError(
                f"Model '{self.model}' isn't available for generateContent.\n"
                f"Set a valid one via --model or GEMINI_MODEL. Available models include:\n"
                f"{preview}"
            )

    async def list_prompts(self) -> list[types.Prompt]:
        """List available prompts from the MCP server (cached)."""
        if self._prompt_listing is None:
            result = await self.client_session.list_prompts()
            self._prompt_listing = result.prompts
        return self._prompt_listing

    async def list_resources(self) -> list[types.Resource]:
        """List available resources from the MCP server (cached)."""
        if self._resource_listing is None:
            result = await self.client_session.list_resources()
            self._resource_listing = result.resources
        return self._resource_listing

    def invalidate_cache(self) -> None:
        """Drop cached prompt/resource listings so the next call re-fetches."""
        self._prompt_listing = None
        self._resource_listing = None
        self._resource_contents.clear()

    async def get_prompt(
        self, name: str, arguments: dict | None = None
    ) -> list[types.PromptMessage]:
        """Retrieve a specific prompt by name, optionally with arguments."""
        result = await self.client_session.get_prompt(name, arguments)
        return result.messages

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the MCP server (cached), returning parsed JSON if applicable."""
        if uri in self._resource_contents:
            return self._resource_contents[uri]
        result = await self.client_session.read_resource(AnyUrl(uri))
        resource = result.contents[0]
        if isinstance(resource, types.TextResourceContents):
            if resource.mimeType == "application/json":
                try:
                    parsed = json.loads(resource.text)
                except Exception:
                    parsed = resource.text
            else:
                parsed = resource.text
        else:
            parsed = str(resource)
        self._resource_contents[uri] = parsed
        return parsed

    async def _extract_resources(self, query: str) -> str:
        """Extract resource mentions from query and fetch their content."""
        mentions = [word[1:] for word in query.split() if word.startswith("@")]
        if not mentions:
            return ""

        resources = await self.list_resources()
        mentioned_docs: list[tuple[str, str]] = []

        for resource in resources:
            # Match against resource names or URIs
            if resource.name in mentions or str(resource.uri) in mentions:
                try:
                    content = await self.read_resource(str(resource.uri))
                    mentioned_docs.append((resource.name, str(content)))
                except Exception as e:
                    logger.warning("Error reading resource %s: %s", resource.name, e)

            # Special handling for dataset list provider
            if "datasets" in resource.name.lower():
                try:
                    items = await self.read_resource(str(resource.uri))
                    if isinstance(items, list):
                        for item in items:
                            if str(item) in mentions:
                                info = await self.client_session.call_tool(
                                    "dataset_info", {"dataset": str(item)}
                                )
                                if not info.isError:
                                    mentioned_docs.append(
                                        (str(item), str(info.content))
                                    )
                except Exception as e:
                    logger.debug("Could not fetch dataset info for a mention: %s", e)

        # Return the collected resources wrapped in XML tags for the LLM
        return "".join(
            f'\n<document name="{name}">\n{content}\n</document>\n'
            for name, content in mentioned_docs
        )

    async def _process_command(self, query: str) -> str | None:
        """Process a command (starting with /) using MCP prompts."""
        if not query.startswith("/"):
            return None

        try:
            words = shlex.split(query)
        except ValueError as e:
            raise ValueError(f"Could not parse command - check your quotes: {e}") from e

        command_name = words[0][1:]
        arg_words = words[1:]

        prompts = await self.list_prompts()
        prompt = next((p for p in prompts if p.name == command_name), None)
        if prompt is None:
            raise ValueError(f"Unknown command: /{command_name}")

        arg_specs = prompt.arguments or []
        if len(arg_words) > len(arg_specs):
            raise ValueError(
                f"/{command_name} takes at most {len(arg_specs)} argument(s), "
                f"got {len(arg_words)}"
            )

        missing = [
            spec.name
            for spec, _ in zip(arg_specs[len(arg_words) :], range(len(arg_specs)))
            if spec.required
        ]
        if missing:
            raise ValueError(
                f"/{command_name} missing required argument(s): {', '.join(missing)}"
            )

        args = {spec.name: word for spec, word in zip(arg_specs, arg_words)}

        messages = await self.get_prompt(command_name, args)
        # Combine MCP prompt messages into a single string for Gemini
        combined_prompt = ""
        for msg in messages:
            if isinstance(msg.content, types.TextContent):
                combined_prompt += f"{msg.role}: {msg.content.text}\n"
            else:
                combined_prompt += f"{msg.role}: {msg.content}\n"
        return combined_prompt

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available MCP tools."""

        # Process a /prompt command if present
        if query.startswith("/"):
            try:
                command_text = await self._process_command(query)
            except Exception as e:
                return f"Error: could not run command '{query.split()[0]}': {e}"
            if command_text is not None:
                query = f"Execute this prompt:\n{command_text}"

        # Inject context from @resource mentions
        added_resources = await self._extract_resources(query)
        if added_resources:
            query = (
                "The following context may be useful:\n"
                "<context>\n"
                f"{added_resources}\n"
                "</context>\n"
                "\n"
                f"User Query: {query}"
            )

        response = await self.chat.send_message(query)

        if not response.text:
            return "Assistant: (no response)"

        return "Assistant: " + response.text
