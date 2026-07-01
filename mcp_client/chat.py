import asyncio
import itertools
import logging
import signal
from collections.abc import Awaitable, Callable
from typing import TypedDict

from mcp.types import Resource
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown

from mcp_client.handlers import GeminiQueryHandler
from mcp_client.nnja_contract import (
    DATASET_LIST_RESOURCE_HINT,
    DATASET_LIST_RESOURCE_URI_PREFIX,
    DATASET_META_LABEL,
)

logger = logging.getLogger(__name__)


class LocalCommand(TypedDict):
    description: str
    handler: Callable[[], Awaitable[None]]


class UnifiedCompleter(Completer):
    """Custom prompt_toolkit completer that handles both prompts (/) and resource mentions (@)."""

    def __init__(self):
        self.prompts = []
        self.resource_items = []  # Store (item, meta) tuples
        self.meta_types = []
        self.local_commands = {}  # name -> {"description": str, "handler": Callable}

    def set_local_commands(self, commands: dict[str, LocalCommand]) -> None:
        self.local_commands = commands

    def update_prompts(self, prompts: list):
        """Update the list of available MCP prompts for / completion."""
        self.prompts = prompts

    def update_resource_items(self, items: list[tuple]):
        """Update the list of available resources and their types for @ completion."""
        self.resource_items = items
        self.meta_types = sorted(set(meta for _, meta in items))

    def get_completions(self, document, complete_event):
        """Decide completion suggestions based on current cursor position."""
        text_before_cursor = document.text_before_cursor

        # Handle @ resource mentions and filtering (e.g. @Dataset:...)
        if "@" in text_before_cursor:
            last_at_pos = text_before_cursor.rfind("@")
            prefix = text_before_cursor[last_at_pos + 1 :]

            # Check for filter delimiter
            type_filter = None
            item_search = prefix

            for d in [":", "/", " "]:
                if d in prefix:
                    parts = prefix.split(d, 1)
                    potential_type = parts[0].lower()
                    for mt in self.meta_types:
                        if mt.lower() == potential_type:
                            type_filter = mt
                            item_search = parts[1]
                            break
                    if type_filter:
                        break

            if type_filter:
                # Provide items filtered by the specified type
                for item, meta in self.resource_items:
                    if meta == type_filter and item.lower().startswith(
                        item_search.lower()
                    ):
                        yield Completion(
                            item,
                            start_position=-len(prefix),
                            display=item,
                            display_meta=meta,
                        )
            else:
                # Provide all items matching the prefix
                for item, meta in self.resource_items:
                    if item.lower().startswith(prefix.lower()):
                        yield Completion(
                            item,
                            start_position=-len(prefix),
                            display=item,
                            display_meta=meta,
                        )

                # Also suggest filters matching the prefix
                for mt in self.meta_types:
                    if mt.lower().startswith(prefix.lower()):
                        yield Completion(
                            f"{mt}:",
                            start_position=-len(prefix),
                            display=f"{mt}:",
                            display_meta="Filter",
                        )
            return

        # Handle / prompt completions
        if text_before_cursor.startswith("/"):
            parts = text_before_cursor[1:].split()

            # Suggest prompt names if at the first word
            if len(parts) <= 1 and not text_before_cursor.endswith(" "):
                cmd_prefix = parts[0] if parts else ""

                for prompt in self.prompts:
                    if prompt.name.startswith(cmd_prefix):
                        yield Completion(
                            prompt.name,
                            start_position=-len(cmd_prefix),
                            display=f"/{prompt.name}",
                            display_meta=prompt.description or "",
                        )

                for name, spec in self.local_commands.items():
                    if name.startswith(cmd_prefix):
                        yield Completion(
                            name,
                            start_position=-len(cmd_prefix),
                            display=f"/{name}",
                            display_meta=spec["description"],
                        )
                return

            # Suggest resource items as arguments after a prompt name
            if len(parts) == 1 and text_before_cursor.endswith(" "):
                for item, meta in self.resource_items:
                    yield Completion(
                        item,
                        start_position=0,
                        display=item,
                        display_meta=meta,
                    )
                return


class CommandAutoSuggest(AutoSuggest):
    """Provides ghost text suggestions for prompt arguments."""

    def __init__(self, prompts: list):
        self.prompt_dict = {prompt.name: prompt for prompt in prompts}

    def get_suggestion(self, buffer: Buffer, document: Document) -> Suggestion | None:
        """Suggest the first argument name when the input is a recognized /command."""
        text = document.text

        if not text.startswith("/"):
            return None

        parts = text[1:].split()

        # If user typed a valid prompt, suggest its first argument name
        if len(parts) == 1 and not text.endswith(" "):
            cmd = parts[0]
            if cmd in self.prompt_dict:
                prompt = self.prompt_dict[cmd]
                if prompt.arguments:
                    return Suggestion(f" {prompt.arguments[0].name}")

        return None


async def _show_thinking(
    message: str = "Assistant is thinking", paused: asyncio.Event | None = None
) -> None:
    """Animate a spinner until cancelled, clearing its own line on exit."""
    blank = "\r" + " " * (len(message) + 6) + "\r"
    was_paused = False
    try:
        for frame in itertools.cycle("|/-\\"):
            is_paused = paused is not None and paused.is_set()
            if is_paused and not was_paused:
                print(blank, end="", flush=True)  # clear once on pausing
            elif not is_paused:
                print(f"\r{message}... {frame}", end="", flush=True)
            was_paused = is_paused
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        # Wipe the spinner line so the response prints cleanly
        print(blank, end="", flush=True)
        raise


class ChatSession:
    """Interactive terminal chat session: prompt loop, completion, and query dispatch."""

    def __init__(
        self, handler: GeminiQueryHandler, pause_spinner: asyncio.Event | None = None
    ) -> None:
        self.handler = handler
        self.mcp = handler.mcp
        # Rendering boundary: prompt_toolkit owns input (the prompt session below:
        # prompts, completion, key bindings); rich owns output (self.console:
        # markdown, status, errors). They share the terminal safely only because
        # they run at disjoint times -- never run a rich Live/Status while a
        # prompt_toolkit prompt is open (that is why the spinner pauses during the
        # elicitation prompt).
        self.console = Console(highlight=False)
        self.completer = UnifiedCompleter()
        self.autosuggester = CommandAutoSuggest([])
        self.local_commands: dict[str, LocalCommand] = {
            "refresh": {
                "description": "Refresh auto-completion suggestions",
                "handler": self._handle_refresh,
            },
        }
        self.completer.set_local_commands(self.local_commands)
        self.session = self._build_session()
        self.pause_spinner = pause_spinner

    async def run(self) -> None:
        self.console.print("\nMCP Client's Chat Started!", style="bold")
        self.console.print(f"Model: {self.handler.model}", style="dim")
        self.console.print(
            "Type your queries. Exit with 'quit', 'q' or Ctrl-D.", style="dim"
        )
        self.console.print(
            "Use @ to mention resources/items and / to use prompts.", style="dim"
        )

        # Initial load of completions
        await self.refresh_completions()

        # Main interaction loop
        while True:
            try:
                query = await self.session.prompt_async("\nYou: ")
                query = query.strip()

                if not query:
                    continue
                if query.lower() in ("quit", "q"):
                    break
                if await self._dispatch_local(query):
                    continue
                await self._respond(query)

            except KeyboardInterrupt:
                self.console.print("\nUse 'quit', 'q', or Ctrl-D to exit.", style="dim")
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"\nError: {str(e)}", style="bold red")

        self.console.print("\nGoodbye!", style="dim")

    async def refresh_completions(self) -> None:
        """Fetch updated prompts and resources from the MCP server to refresh autocompletion."""
        try:
            # Fetch prompts and resources in parallel
            prompts, resources = await asyncio.gather(
                self.mcp.list_prompts(),
                self.mcp.list_resources(),
            )

            # Update prompts
            self.completer.update_prompts(prompts)
            self.autosuggester.prompt_dict = {p.name: p for p in prompts}

            # Update resources and nested items from list providers
            all_items = []

            for res in resources:
                meta = self.get_meta_for_resource(res)
                is_list_provider = (
                    "list" in res.name.lower()
                    or DATASET_LIST_RESOURCE_HINT in res.name.lower()
                )

                # If it's a list provider, fetch the items inside it
                if is_list_provider:
                    try:
                        content = await self.mcp.read_resource(str(res.uri))

                        # Items inside a list provider:
                        #   an uncategorized provider's items are generic "Item"s
                        #   a categorized item (e.g. "Dataset") keeps its name
                        item_meta = "Item" if meta == "Resource" else meta

                        if isinstance(content, list):
                            all_items.extend([(str(i), item_meta) for i in content])
                        elif isinstance(content, str) and "\n" in content:
                            all_items.extend(
                                [
                                    (line.strip(), item_meta)
                                    for line in content.split("\n")
                                    if line.strip()
                                ]
                            )
                    except Exception as e:
                        logger.debug(
                            "Skipping list provider %s during completion refresh: %s",
                            res.name,
                            e,
                        )
                else:
                    # Add the resource name itself only if it's not a list provider
                    all_items.append((res.name, meta))

            # Deduplicate items while preserving metadata
            self.completer.update_resource_items(sorted(set(all_items)))
        except Exception as e:
            self.console.print(
                f"Warning: Could not refresh completions: {e}", style="yellow"
            )

    async def _dispatch_local(self, query: str) -> bool:
        """Run a client-side command if `query` names one. Return True if handled."""
        if not query.startswith("/"):
            return False

        parts = query[1:].split()
        if not parts:  # a bare "/" (or "/" + whitespace) names no command
            return False
        command = parts[0].lower()
        spec = self.local_commands.get(command)
        if spec is None:
            return False
        await spec["handler"]()
        return True

    async def _respond(self, query: str) -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(self._run_query(query))

        def _cancel_on_sigint(*_):
            loop.call_soon_threadsafe(task.cancel)  # safe to call from a signal handler

        previous = signal.signal(signal.SIGINT, _cancel_on_sigint)

        try:
            result = await task
            if result.startswith("Error:"):
                self.console.print(f"\n{result}", style="bold red")
            else:
                self.console.print("\nAssistant:", style="bold cyan")
                self.console.print(Markdown(result))
        except asyncio.CancelledError:
            self.console.print(
                "\n(cancelled - 'quit', 'q', or Ctrl-D to exit)", style="dim"
            )
        finally:
            signal.signal(signal.SIGINT, previous)  # restore for the idle prompt

    async def _run_query(self, query: str) -> str:
        spinner = asyncio.create_task(_show_thinking(paused=self.pause_spinner))
        try:
            # Process the query through the handler and MCP
            return await self.handler.process_query(query)
        finally:
            spinner.cancel()
            try:
                # Let the cancellation propagate and clean up
                await spinner
            except asyncio.CancelledError:
                pass

    async def _handle_refresh(self) -> None:
        self.mcp.invalidate_cache()
        await self.refresh_completions()
        self.console.print("Completions refreshed!", style="dim")

    def _build_session(self) -> PromptSession:
        return PromptSession(
            completer=self.completer,
            history=InMemoryHistory(),
            key_bindings=self._build_key_bindings(),
            style=Style.from_dict(
                {
                    "prompt": "bold ansigreen",
                    "completion-menu.completion": "bg:#222222 #ffffff",
                    "completion-menu.completion.current": "bg:#444444 #ffffff",
                }
            ),
            complete_while_typing=True,
            auto_suggest=self.autosuggester,
        )

    @staticmethod
    def _build_key_bindings() -> KeyBindings:
        kb = KeyBindings()

        @kb.add("/")
        def _(event):
            """Open completion menu when / is typed at the start of a line."""
            buffer = event.app.current_buffer
            buffer.insert_text("/")
            if buffer.document.is_cursor_at_the_end and buffer.text == "/":
                buffer.start_completion(select_first=False)

        @kb.add("@")
        def _(event):
            """Open completion menu immediately when @ is typed."""
            buffer = event.app.current_buffer
            buffer.insert_text("@")
            if buffer.document.is_cursor_at_the_end:
                buffer.start_completion(select_first=False)

        return kb

    @staticmethod
    def get_meta_for_resource(res: Resource) -> str:
        """Categorize resources by their meta type for UI display."""
        if str(res.uri).startswith(DATASET_LIST_RESOURCE_URI_PREFIX):
            return DATASET_META_LABEL
        return "Resource"


async def run_chat(
    handler: GeminiQueryHandler, pause_spinner: asyncio.Event | None = None
) -> None:
    """Create a ChatSession for `handler` and run its interactive loop."""
    await ChatSession(handler, pause_spinner).run()
