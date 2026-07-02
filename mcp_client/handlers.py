import logging
import os
import re
import shlex

import mcp.types as types
from dotenv import load_dotenv
from fuzzywuzzy import process
from google import genai

from mcp_client.gateway import MCPGateway
from mcp_client.nnja_contract import (
    DATASET_INFO_ARG,
    DATASET_INFO_TOOL,
    DATASET_LIST_RESOURCE_HINT,
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

# Trailing punctuation attached to an @-mention (e.g. "@datasets." or "@datasets,").
# Resource names/URIs and dataset IDs all *end* in an alphanumeric ("datasets",
# "data://datasets", "conv-adpsfc-NC000007"), so a trailing run of non-alphanumerics is
# always punctuation the user appended, never part of the identifier. (Punctuation
# *before* the "@" is a separate case the completer never produces; not handled here.)
_MENTION_TRAILING_PUNCT = re.compile(r"[^A-Za-z0-9]+$")


def _extract_mentions(query: str) -> set[str]:
    """Return the @-mentioned resource/dataset names in `query`, trailing punctuation stripped.

    Splits on whitespace, keeps the `@...` tokens, drops the leading `@`, and strips any
    trailing punctuation so hand-typed mentions like "@datasets." still match a resource
    named "datasets". Empty results (a bare "@") are dropped.
    """
    return {
        stripped
        for word in query.split()
        if word.startswith("@")
        and (stripped := _MENTION_TRAILING_PUNCT.sub("", word[1:]))
    }


class GeminiQueryHandler:
    """Handle Gemini API interaction and MCP tool execution."""

    def __init__(self, mcp: MCPGateway, model: str | None = None):
        self.mcp = mcp
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
                tools=[mcp.client_session],  # Expose MCP tools to the LLM
            ),
        )

    async def verify_model(self) -> None:
        """Fail fast if the configured model can't serve generateContent."""
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

    async def _extract_resources(self, query: str) -> str:
        """Extract resource mentions from query and fetch their content."""
        mentions = _extract_mentions(query)
        if not mentions:
            return ""

        resources = await self.mcp.list_resources()
        mentioned_docs: list[tuple[str, str]] = []

        for resource in resources:
            # Match against resource names or URIs
            if resource.name in mentions or str(resource.uri) in mentions:
                try:
                    content = await self.mcp.read_resource(str(resource.uri))
                    mentioned_docs.append((resource.name, str(content)))
                    mentions.discard(resource.name)
                    mentions.discard(str(resource.uri))
                except Exception as e:
                    logger.warning("Error reading resource %s: %s", resource.name, e)

        # If mentions are left, see if they're in the datasets list
        if mentions:
            ds_res = next(
                (r for r in resources if DATASET_LIST_RESOURCE_HINT in r.name.lower()),
                None,
            )
            if ds_res:
                try:
                    items = await self.mcp.read_resource(str(ds_res.uri))
                    if isinstance(items, list):
                        for item in items:
                            if str(item) in mentions:
                                info = await self.mcp.client_session.call_tool(
                                    DATASET_INFO_TOOL, {DATASET_INFO_ARG: str(item)}
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

        prompts = await self.mcp.list_prompts()
        prompt = next((p for p in prompts if p.name == command_name), None)
        if prompt is None:
            raise ValueError(f"Unknown command: /{command_name}")

        arg_specs = prompt.arguments or []
        if len(arg_words) > len(arg_specs):
            raise ValueError(
                f"/{command_name} takes at most {len(arg_specs)} argument(s), "
                f"got {len(arg_words)}"
            )

        missing = [spec.name for spec in arg_specs[len(arg_words) :] if spec.required]
        if missing:
            raise ValueError(
                f"/{command_name} missing required argument(s): {', '.join(missing)}"
            )

        args = {spec.name: word for spec, word in zip(arg_specs, arg_words)}

        messages = await self.mcp.get_prompt(command_name, args)
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
        original_query = query

        # Process a /prompt command if present
        if query.startswith("/"):
            try:
                command_text = await self._process_command(query)
            except Exception as e:
                return f"Error: could not run command '{query.split()[0]}': {e}"
            if command_text is not None:
                query = f"Execute this prompt:\n{command_text}"

        # Inject context from @resource mentions in the original query
        added_resources = await self._extract_resources(original_query)
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
            reason = None
            if response.candidates:
                reason = response.candidates[0].finish_reason
            logger.debug("No text (finish_reason=%s): %s", reason, response)
            return f"(no response{f' - {reason}' if reason else ''})"

        return response.text
