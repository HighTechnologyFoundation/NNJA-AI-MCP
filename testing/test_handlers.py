"""Unit tests for the GeminiQueryHandler query pipeline (no Gemini, no network).

`bare_handler` builds the handler with only `.mcp` set, so `_extract_resources` and
`_process_command` are exercised against a stub gateway with no Gemini client at all.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from _fakes import bare_handler, prompt, resource, stub_gateway, sync, text_message

# _extract_resources


@sync
async def test_extract_resources_no_mentions_returns_empty():
    gw = stub_gateway()
    handler = bare_handler(gw)

    assert await handler._extract_resources("just a plain query") == ""
    gw.list_resources.assert_not_called()  # short-circuits before touching the server


@sync
async def test_extract_resources_matches_resource_by_name():
    gw = stub_gateway(
        resources=[resource("schema", "data://schema")], read="SCHEMA TEXT"
    )
    handler = bare_handler(gw)

    out = await handler._extract_resources("tell me about @schema")

    assert '<document name="schema">' in out
    assert "SCHEMA TEXT" in out


@sync
async def test_extract_resources_dataset_mention_calls_tool():
    gw = stub_gateway(
        resources=[resource("datasets", "data://datasets")],
        read=["ADPSFC", "OTHER"],
        tool_result=SimpleNamespace(isError=False, content="DATASET INFO"),
    )
    handler = bare_handler(gw)

    out = await handler._extract_resources("look at @ADPSFC")

    assert '<document name="ADPSFC">' in out
    assert "DATASET INFO" in out
    gw.client_session.call_tool.assert_awaited_once_with(
        "dataset_info", {"dataset": "ADPSFC"}
    )


@sync
async def test_extract_resources_skips_errored_tool():
    gw = stub_gateway(
        resources=[resource("datasets", "data://datasets")],
        read=["ADPSFC"],
        tool_result=SimpleNamespace(isError=True, content="boom"),
    )
    handler = bare_handler(gw)

    assert await handler._extract_resources("look at @ADPSFC") == ""


# _process_command


@sync
async def test_process_command_unknown_raises():
    gw = stub_gateway(prompts=[prompt("cite", [("dataset", True)])])
    handler = bare_handler(gw)

    with pytest.raises(ValueError, match="Unknown command"):
        await handler._process_command("/nope")


@sync
async def test_process_command_too_many_args_raises():
    gw = stub_gateway(prompts=[prompt("cite", [("dataset", True)])])
    handler = bare_handler(gw)

    with pytest.raises(ValueError, match="at most 1 argument"):
        await handler._process_command("/cite a b")


@sync
async def test_process_command_missing_required_raises():
    gw = stub_gateway(prompts=[prompt("cite", [("dataset", True)])])
    handler = bare_handler(gw)

    with pytest.raises(ValueError, match="missing required argument"):
        await handler._process_command("/cite")


@sync
async def test_process_command_combines_prompt_messages():
    gw = stub_gateway(
        prompts=[prompt("cite", [("dataset", True)])],
        messages=[text_message("user", "Cite mydataset")],
    )
    handler = bare_handler(gw)

    result = await handler._process_command("/cite mydataset")

    assert result == "user: Cite mydataset\n"
    gw.get_prompt.assert_awaited_once_with("cite", {"dataset": "mydataset"})


@sync
async def test_process_command_bad_quotes_raises():
    gw = stub_gateway(prompts=[prompt("cite", [("dataset", True)])])
    handler = bare_handler(gw)

    with pytest.raises(ValueError, match="check your quotes"):
        await handler._process_command('/cite "unterminated')


# process_query response handling


@sync
async def test_process_query_returns_text_when_present():
    handler = bare_handler(stub_gateway())
    handler.chat = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(text="the answer"))
    )

    assert await handler.process_query("hello") == "Assistant: the answer"


@sync
async def test_process_query_empty_response_surfaces_finish_reason(caplog):
    handler = bare_handler(stub_gateway())
    response = SimpleNamespace(
        text="", candidates=[SimpleNamespace(finish_reason="SAFETY")]
    )
    handler.chat = SimpleNamespace(send_message=AsyncMock(return_value=response))

    with caplog.at_level(logging.DEBUG, logger="mcp_client.handlers"):
        result = await handler.process_query("hello")

    assert result == "Assistant: (no response - SAFETY)"
    assert any("SAFETY" in r.getMessage() for r in caplog.records)  # cause is logged


@sync
async def test_process_query_empty_response_without_candidates():
    handler = bare_handler(stub_gateway())
    response = SimpleNamespace(text="", candidates=[])
    handler.chat = SimpleNamespace(send_message=AsyncMock(return_value=response))

    assert await handler.process_query("hello") == "Assistant: (no response)"
