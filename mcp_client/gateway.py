import json
from typing import Any

import mcp.types as types
from pydantic import AnyUrl
from mcp import ClientSession


class MCPGateway:
    """Cached access to the MCP server: listings, reads, prompt fetch, tool calls."""
    def __init__(self, client_session: ClientSession):
        self.client_session = client_session
        self._prompt_listing: list[types.Prompt] | None = None
        self._resource_listing: list[types.Resource] | None = None
        self._resource_contents: dict[str, Any] = {}

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

    async def get_prompt(
        self, name: str, arguments: dict | None = None
    ) -> list[types.PromptMessage]:
        """Retrieve a specific prompt by name, optionally with arguments."""
        result = await self.client_session.get_prompt(name, arguments)
        return result.messages

    async def read_resource(self, uri):
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

    def invalidate_cache(self) -> None:
        """Drop cached prompt/resource listings so the next call re-fetches."""
        self._prompt_listing = None
        self._resource_listing = None
        self._resource_contents.clear()

    async def call_tool(self, name, args):
        """Simple function to call a tool through the client session."""
        return await self.client_session.call_tool(name, args)