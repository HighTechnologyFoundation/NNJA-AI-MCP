import asyncio
import itertools
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

from mcp_client.handlers import GeminiQueryHandler


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
        self.prompts = prompts
        self.prompt_dict = {prompt.name: prompt for prompt in prompts}

    def get_suggestion(self, buffer: Buffer, document: Document) -> Suggestion | None:
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


async def _show_thinking(message: str = "Assistant is thinking") -> None:
    """Animate a spinner until cancelled. Clears its own line on exit."""
    try:
        for frame in itertools.cycle("|/-\\"):
            print(f"\r{message}... {frame}", end="", flush=True)
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        # Wipe the spinner line so the response prints cleanly
        print("\r" + " " * (len(message) + 6) + "\r", end="", flush=True)
        raise


class ChatSession:
    def __init__(self, handler: GeminiQueryHandler) -> None:
        self.handler = handler
        self.mcp = handler.mcp
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

    async def run(self) -> None:
        print("\nMCP Client's Chat Started!")
        print(f"Model: {self.handler.model}")
        print("Type your queries. Exit with 'quit', 'q' or Ctrl-D.")
        print("Use @ to mention resources/items and / to use prompts.")

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
                print("\nUse 'quit', 'q', or Ctrl-D to exit.")
                continue
            except EOFError:
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

        print("\nGoodbye!")

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
            self.autosuggester.prompts = prompts
            self.autosuggester.prompt_dict = {p.name: p for p in prompts}

            # Update resources and nested items from list providers
            all_items = []

            for res in resources:
                meta = self.get_meta_for_resource(res)
                is_list_provider = (
                    "list" in res.name.lower() or "datasets" in res.name.lower()
                )

                # If it's a list provider, fetch the items inside it
                if is_list_provider:
                    try:
                        content = await self.mcp.read_resource(str(res.uri))
                        item_meta = meta.rstrip("s") if meta.endswith("s") else meta
                        if item_meta == "Resource":
                            item_meta = "Item"

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
                    except Exception:
                        pass
                else:
                    # Add the resource name itself only if it's not a list provider
                    all_items.append((res.name, meta))

            # Deduplicate items while preserving metadata
            self.completer.update_resource_items(sorted(set(all_items)))
        except Exception as e:
            print(f"Warning: Could not refresh completions: {e}")

    async def _dispatch_local(self, query: str) -> bool:
        """Run a client-side command if `query` names one. Returns True is handled."""
        if not query.startswith("/"):
            return False

        command = query[1:].split()[0].lower()
        spec = self.local_commands.get(command)
        if spec is None:
            return False
        await spec["handler"]()
        return True

    async def _respond(self, query: str) -> None:
        spinner = asyncio.create_task(_show_thinking())
        try:
            # Process the query through the handler and MCP
            response = await self.handler.process_query(query)
        finally:
            spinner.cancel()
            try:
                # Let the cancellation propogate and clean up
                await spinner
            except asyncio.CancelledError:
                pass

        print("\n" + response)

    async def _handle_refresh(self) -> None:
        self.mcp.invalidate_cache()
        await self.refresh_completions()
        print("Completions refreshed!")

    def _build_session(self) -> PromptSession:
        return PromptSession(
            completer=self.completer,
            history=InMemoryHistory(),
            key_bindings=self._build_key_bindings(),
            style=Style.from_dict(
                {
                    "prompt": "#aaaaaa",
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
        if str(res.uri).startswith("data://datasets"):
            return "Dataset"
        return "Resource"


async def run_chat(handler: GeminiQueryHandler) -> None:
    await ChatSession(handler).run()
