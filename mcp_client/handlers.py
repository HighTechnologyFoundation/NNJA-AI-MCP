import json
import os
from typing import Any

import mcp.types as types
from dotenv import load_dotenv
from google import genai
from mcp import ClientSession
from pydantic import AnyUrl

# Load environment variables from .env file
load_dotenv()


class GeminiQueryHandler:
    """Handle Gemini API interaction and MCP tool execution."""

    def __init__(self, client_session: ClientSession):
        self.client_session = client_session
        if not (api_key := os.getenv("GEMINI_API_KEY")):
            raise RuntimeError(
                "Error: GEMINI_API_KEY environment variable not set",
            )

        # Initialize the Gemini client
        self.gemini = genai.Client(api_key=api_key)

        # Create an asynchronous chat session with MCP tools enabled for the model
        self.chat = self.gemini.aio.chats.create(
            model="gemini-3.1-flash-lite",
            config=genai.types.GenerateContentConfig(
                tools=[self.client_session],  # Expose MCP tools to the LLM
            ),
        )

    async def list_prompts(self) -> list[types.Prompt]:
        """List available prompts from the MCP server."""
        result = await self.client_session.list_prompts()
        return result.prompts

    async def list_resources(self) -> list[types.Resource]:
        """List available resources from the MCP server."""
        result = await self.client_session.list_resources()
        return result.resources

    async def get_prompt(
        self, name: str, arguments: dict | None = None
    ) -> list[types.PromptMessage]:
        """Retrieve a specific prompt by name, optionally with arguments."""
        result = await self.client_session.get_prompt(name, arguments)
        return result.messages

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the MCP server and returns parsed JSON if applicable."""
        result = await self.client_session.read_resource(AnyUrl(uri))
        resource = result.contents[0]
        if isinstance(resource, types.TextResourceContents):
            if resource.mimeType == "application/json":
                try:
                    return json.loads(resource.text)
                except Exception:
                    return resource.text
            return resource.text
        return str(resource)

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
                    print(f"Error reading resource {resource.name}: {e}")

            # Special handling for list providers
            if "list" in resource.name.lower() or "datasets" in resource.name.lower():
                try:
                    items = await self.read_resource(str(resource.uri))
                    if isinstance(items, list):
                        for item in items:
                            if str(item) in mentions:
                                # If an item matches, we might want to fetch IT as a resource
                                # OR just use the item name as context.
                                # Often there's a template, but without knowing it, we'll just
                                # include the fact that this item was selected.
                                mentioned_docs.append(
                                    (str(item), f"Selected item from {resource.name}")
                                )
                except Exception:
                    pass

        # Return the collected resources wrapped in XML tags for the LLM
        return "".join(
            f'\n<document name="{name}">\n{content}\n</document>\n'
            for name, content in mentioned_docs
        )

    async def _process_command(self, query: str) -> str:
        """Process a command (starting with /) using MCP prompts."""
        if not query.startswith("/"):
            return ""

        words = query.split()
        command_name = words[0][1:]
        arg_words = words[1:]
        args = {}

        prompts = await self.list_prompts()
        prompt = next((p for p in prompts if p.name == command_name), None)
        if prompt and prompt.arguments:
            args = {arg.name: word for arg, word in zip(prompt.arguments, arg_words)}

        try:
            messages = await self.get_prompt(command_name, args)
            # Combine MCP prompt messages into a single string for Gemini
            combined_prompt = ""
            for msg in messages:
                if isinstance(msg.content, types.TextContent):
                    combined_prompt += f"{msg.role}: {msg.content.text}\n"
                else:
                    combined_prompt += f"{msg.role}: {msg.content}\n"
            return combined_prompt
        except Exception as e:
            return f"Error processing command {command_name}: {e}"

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available MCP tools."""

        # Process a /prompt command if present
        if query.startswith("/"):
            command_text = await self._process_command(query)
            if command_text:
                query = f"Execute this prompt:\n{command_text}"

        # Inject context from @resource mentions
        added_resources = await self._extract_resources(query)
        if added_resources:
            query = f"""
            The following context may be useful:
            <context>
            {added_resources}
            </context>

            User Query: {query}
            """

        response = await self.chat.send_message(query)

        if not response.text:
            return "Assistant: (no response)"

        return "Assistant: " + response.text
