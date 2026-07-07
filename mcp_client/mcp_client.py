import asyncio
import pathlib
import re
import signal
import sys
from contextlib import AsyncExitStack, suppress
from typing import Any, Self

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.table import Table
from rich.text import Text

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

        # Pauses spinner while this is set
        self._elicitation_active = asyncio.Event()

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
        except BaseException as e:
            # A failed connect surfaces messily: over HTTP a down server bubbles up as a
            # CancelledError from the transport's task group (not an Exception), and any
            # partially-entered contexts are left on the stack because __aexit__ isn't
            # called when __aenter__ raises. Close them here (suppressing teardown noise),
            # let genuine interrupts through, then raise one friendly, actionable error.
            with suppress(BaseException):
                await self.exit_stack.aclose()
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            if self.use_http:
                # Server-not-running is the likeliest HTTP connect failure at a demo;
                # mirror the actionable hint the example scripts give (client.py /
                # testing/_client.py). The propagated CancelledError carries no useful
                # "connection refused" text, so appending it would only add noise.
                raise RuntimeError(
                    f"Could not reach the MCP server at {self.target}. "
                    "Is it running? Start it in HTTP mode first:\n"
                    "- PowerShell: $env:MCP_TRANSPORT='http'; uv run server.py\n"
                    "- bash/zsh:   MCP_TRANSPORT=http uv run server.py\n"
                    "- Docker:     docker build -t nnja-ai-mcp .\n"
                    "              docker run -p 8000:8000 nnja-ai-mcp"
                ) from e
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e

    async def _handle_elicitation(
        self, context: Any, params: types.ElicitRequestParams
    ) -> types.ElicitResult:
        """Confirm a server-initiated elicitation (e.g. an oversized data load).

        The server pauses its tool call to ask the human before proceeding; "y"/"yes"
        accepts and lets the work run, anything else declines so the server can abort.
        Ctrl-C / Ctrl-D cancels the whole turn (like Ctrl-C during a running query) and
        returns a "cancel" response; without catching it here the KeyboardInterrupt
        would crash the session's receive loop this callback runs on.
        """
        self._elicitation_active.set()

        # FormattedText (style, text) tuples rather than HTML: params.message is
        # server-supplied and could contain markup characters that break HTML parsing.
        prompt = FormattedText(
            [
                ("bold ansiyellow", f"\nWARNING: {params.message}\n"),
                ("bold", "Proceed? [y/N] "),
            ]
        )
        try:
            with patch_stdout():
                answer = await PromptSession().prompt_async(prompt)
        except (KeyboardInterrupt, EOFError):
            # Cancel the whole turn, like Ctrl-C during a running query: re-raise SIGINT
            # so _respond's handler cancels the query task (prompt_toolkit restores that
            # handler before raising here). Still return a response so the server, which
            # is awaiting this elicitation, isn't left hanging.
            signal.raise_signal(signal.SIGINT)
            return types.ElicitResult(action="cancel")
        finally:
            self._elicitation_active.clear()

        if answer.strip().lower() in {"y", "yes"}:
            return types.ElicitResult(action="accept")
        return types.ElicitResult(action="decline")

    async def list_all_members(self) -> None:
        """List all server-side tools, prompts, and resources."""
        console = Console()
        sections = {
            "tools": self.client_session.list_tools,
            "prompts": self.client_session.list_prompts,
            "resources": self.client_session.list_resources,
        }
        for section, listing_method in sections.items():
            try:
                items = getattr(await listing_method(), section)
            except Exception as e:
                console.print(Text(f"{section.upper()}: Error - {e}", style="red"))
                continue
            if not items:
                console.print(Text(f"{section.upper()}: none available", style="dim"))
                continue

            table = Table(
                title=f"{section.upper()} ({len(items)})",
                title_justify="left",
                title_style="bold",
            )
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("Description")
            for item in items:
                # get summary from docstring (first paragraph)
                summary = (item.description or "").strip().split("\n\n")[0]
                table.add_row(item.name, Text(summary or "No description"))

            console.print(table)
            console.print()

    async def run_chat(self, model: str | None = None) -> None:
        """Initialize the query handler and launch the interactive chat UI."""
        try:
            mcp = MCPGateway(self.client_session)
            handler = GeminiQueryHandler(mcp, model=model)
            await handler.verify_model()
            await chat.run_chat(handler, pause_spinner=self._elicitation_active)
        except RuntimeError as e:
            print(e)
