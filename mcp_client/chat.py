from typing import List, Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.buffer import Buffer


class UnifiedCompleter(Completer):
    def __init__(self):
        self.prompts = []
        self.prompt_dict = {}
        self.resource_items = []  # Store (item, meta) tuples
        self.meta_types = []

    def update_prompts(self, prompts: List):
        self.prompts = prompts
        self.prompt_dict = {prompt.name: prompt for prompt in prompts}

    def update_resource_items(self, items: List[tuple]):
        self.resource_items = items
        self.meta_types = sorted(list(set(meta for _, meta in items)))

    def get_completions(self, document, complete_event):
        text_before_cursor = document.text_before_cursor

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
                # Filtered items
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
                # Regular items matching prefix
                for item, meta in self.resource_items:
                    if item.lower().startswith(prefix.lower()):
                        yield Completion(
                            item,
                            start_position=-len(prefix),
                            display=item,
                            display_meta=meta,
                        )

                # Also suggest filters
                for mt in self.meta_types:
                    if mt.lower().startswith(prefix.lower()):
                        yield Completion(
                            f"{mt}:",
                            start_position=-len(prefix),
                            display=f"{mt}:",
                            display_meta="Filter",
                        )
            return

        if text_before_cursor.startswith("/"):
            parts = text_before_cursor[1:].split()

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
                return

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
    def __init__(self, prompts: List):
        self.prompts = prompts
        self.prompt_dict = {prompt.name: prompt for prompt in prompts}

    def get_suggestion(
        self, buffer: Buffer, document: Document
    ) -> Optional[Suggestion]:
        text = document.text

        if not text.startswith("/"):
            return None

        parts = text[1:].split()

        if len(parts) == 1 and not text.endswith(" "):
            cmd = parts[0]
            if cmd in self.prompt_dict:
                prompt = self.prompt_dict[cmd]
                if prompt.arguments:
                    return Suggestion(f" {prompt.arguments[0].name}")

        return None


async def run_chat(handler) -> None:
    """Run an AI-handled chat session with autocompletion."""
    print("\nMCP Client's Chat Started!")
    print("Type your queries or 'quit' to exit.")
    print("Use @ to mention resources/items and / to use prompts.")

    completer = UnifiedCompleter()
    autosuggester = CommandAutoSuggest([])

    async def refresh_completions():
        try:
            prompts = await handler.list_prompts()
            completer.update_prompts(prompts)
            autosuggester.prompts = prompts
            autosuggester.prompt_dict = {p.name: p for p in prompts}

            resources = await handler.list_resources()
            all_items = []

            def get_meta_for_resource(res_name: str) -> str:
                name_lower = res_name.lower()
                if "dataset" in name_lower:
                    return "Dataset"
                if "file" in name_lower:
                    return "File"
                if "log" in name_lower:
                    return "Log"
                return "Resource"

            for res in resources:
                meta = get_meta_for_resource(res.name)
                is_list_provider = (
                    "list" in res.name.lower() or "datasets" in res.name.lower()
                )

                # If it's a list provider, fetch and add its items
                if is_list_provider:
                    try:
                        content = await handler.read_resource(str(res.uri))
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
                    except:
                        pass
                else:
                    # Add the resource name itself only if it's not a list provider
                    all_items.append((res.name, meta))

            # Deduplicate items while preserving metadata
            completer.update_resource_items(sorted(list(set(all_items))))
        except Exception as e:
            print(f"Warning: Could not refresh completions: {e}")

    # Initial load
    await refresh_completions()
    kb = KeyBindings()

    @kb.add("/")
    def _(event):
        buffer = event.app.current_buffer
        if buffer.document.is_cursor_at_the_end and not buffer.text:
            buffer.insert_text("/")
            buffer.start_completion(select_first=False)
        else:
            buffer.insert_text("/")

    @kb.add("@")
    def _(event):
        buffer = event.app.current_buffer
        buffer.insert_text("@")
        if buffer.document.is_cursor_at_the_end:
            buffer.start_completion(select_first=False)

    session = PromptSession(
        completer=completer,
        history=InMemoryHistory(),
        key_bindings=kb,
        style=Style.from_dict(
            {
                "prompt": "#aaaaaa",
                "completion-menu.completion": "bg:#222222 #ffffff",
                "completion-menu.completion.current": "bg:#444444 #ffffff",
            }
        ),
        complete_while_typing=True,
        auto_suggest=autosuggester,
    )

    while True:
        try:
            query = await session.prompt_async("\nYou: ")
            query = query.strip()

            if not query:
                continue
            if query.lower() in ("quit", "q"):
                break

            response = await handler.process_query(query)
            print("\n" + response)

            # Refresh in case server state changed
            await refresh_completions()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

    print("\nGoodbye!")
