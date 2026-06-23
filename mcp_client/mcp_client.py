import pathlib
import re
import sys
from contextlib import AsyncExitStack
from typing import Any, Awaitable, Callable, Self

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from mcp_client import chat
from mcp_client.gateway import MCPGateway
from mcp_client.handlers import GeminiQueryHandler


class MCPClient:
    """Terminal-based MCP client to interact with an MCP server."""

    # Store the target (server script path or URL) and set up an exit stack
    def __init__(self, target: str):
        self.target = target
        self.use_http = target.startswith(("http://", "https://"))
        self.client_session: ClientSession

        # Simplifies managing multiple async context managers
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        """Establish the server connection when entering `async with`."""
        self.client_session = await self._connect_to_server()
        return self

    async def __aexit__(self, *_) -> None:
        """Clean up the server connection and exit stack upon exit."""
        await self.exit_stack.aclose()

    async def _connect_to_server(self) -> ClientSession:
        """Connect to the MCP server and initialize the MCP ClientSession.

        The server connection is handled depending on the mode:
            - In stdio mode, it spawns the MCP server as a subprocess.
            - In HTTP mode, it attaches to an already-running server.
        """
        if not self.use_http and not pathlib.Path(self.target).exists():
            if re.match(r"[\w.\-]+:\d+", self.target):
                raise RuntimeError(
                    f"'{self.target}' looks like a server address. "
                    f"Did you mean 'http://{self.target}'?"
                )
            raise RuntimeError(f"MCP server script '{self.target}' not found")
        try:
            if self.use_http:
                read, write, _ = await self.exit_stack.enter_async_context(
                    streamable_http_client(self.target)
                )
            else:
                read, write = await self.exit_stack.enter_async_context(
                    stdio_client(
                        server=StdioServerParameters(
                            command=sys.executable,
                            args=[self.target],
                            env=None,
                        )
                    )
                )

            # Create the MCP session over the chosen stream. Supplying an elicitation
            # callback both handles server elicitation requests and advertises the
            # elicitation capability during the initialization handshake.
            client_session = await self.exit_stack.enter_async_context(
                ClientSession(
                    read, write, elicitation_callback=self._handle_elicitation
                )
            )

            # Perform MCP initialization handshake
            await client_session.initialize()
            return client_session
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e

    async def _handle_elicitation(
        self, context: Any, params: types.ElicitRequestParams
    ) -> types.ElicitResult:
        """Confirm a server-initiated elicitation (e.g. an oversized data load).

        The server pauses its tool call to ask the human before proceeding; "y"/"yes"
        accepts and lets the work run, anything else declines so the server can abort.
        """
        with patch_stdout():
            answer = await PromptSession().prompt_async(
                f"\nWARNING: {params.message}\nProceed? [y/N] "
            )
        if answer.strip().lower() in {"y", "yes"}:
            return types.ElicitResult(action="accept")
        return types.ElicitResult(action="decline")

    async def list_all_members(self) -> None:
        """List all server-side tools, prompts, and resources."""
        print("MCP Server Members")
        print("=" * 50)

        sections = {
            "tools": self.client_session.list_tools,
            "prompts": self.client_session.list_prompts,
            "resources": self.client_session.list_resources,
        }
        for section, listing_method in sections.items():
            await self._list_section(section, listing_method)

        print("\n" + "=" * 50)

    async def _list_section(
        self,
        section: str,
        list_method: Callable[[], Awaitable[Any]],
    ) -> None:
        """Fetch and print details for a specific section (tools/prompts/resources)."""
        try:
            items = getattr(await list_method(), section)
            if items:
                print(f"\n{section.upper()} ({len(items)}):")
                print("-" * 30)
                for item in items:
                    description = item.description or "No description"
                    print(f" > {item.name} - {description}\n")
            else:
                print(f"\n{section.upper()}: None available")
        except Exception as e:
            print(f"\n{section.upper()}: Error - {e}")

    async def run_chat(self, model: str | None = None) -> None:
        """Initialize the query handler and launch the interactive chat UI."""
        try:
            mcp = MCPGateway(self.client_session)
            handler = GeminiQueryHandler(mcp, model=model)
            await handler.verify_model()
            await chat.run_chat(handler)
        except RuntimeError as e:
            print(e)
