"""Unit tests for MCPGateway: caching and read_resource content parsing.

No live server, no Gemini, no network — a fake ClientSession drives every case.
"""

from _fakes import blob_contents, fake_session, sync, text_contents

from mcp_client.gateway import MCPGateway

# Caching


@sync
async def test_list_prompts_is_cached():
    session = fake_session(prompts=["p1"])
    gw = MCPGateway(session)

    first = await gw.list_prompts()
    second = await gw.list_prompts()

    assert first == second == ["p1"]
    assert session.list_prompts.call_count == 1  # fetched once, then served from cache


@sync
async def test_list_resources_is_cached():
    session = fake_session(resources=["r1"])
    gw = MCPGateway(session)

    first = await gw.list_resources()
    second = await gw.list_resources()

    assert first == second == ["r1"]
    assert session.list_resources.call_count == 1


@sync
async def test_read_resource_caches_per_uri():
    calls = {"n": 0}

    def reader(_uri):
        calls["n"] += 1
        return text_contents('{"call": %d}' % calls["n"])

    session = fake_session(read_side_effect=reader)
    gw = MCPGateway(session)

    a1 = await gw.read_resource("data://a")
    a2 = await gw.read_resource("data://a")  # cached -> no new fetch
    b1 = await gw.read_resource("data://b")  # new uri -> fetched

    assert a1 == a2 == {"call": 1}
    assert b1 == {"call": 2}
    assert session.read_resource.call_count == 2


@sync
async def test_invalidate_cache_forces_refetch_of_all():
    session = fake_session(
        prompts=["p"], resources=["r"], read_return=text_contents("1")
    )
    gw = MCPGateway(session)

    await gw.list_prompts()
    await gw.list_resources()
    await gw.read_resource("data://a")

    gw.invalidate_cache()

    await gw.list_prompts()
    await gw.list_resources()
    await gw.read_resource("data://a")

    assert session.list_prompts.call_count == 2
    assert session.list_resources.call_count == 2
    assert session.read_resource.call_count == 2


# read_resource content parsing


@sync
async def test_read_resource_parses_json():
    session = fake_session(
        read_return=text_contents('{"a": 1, "b": [2, 3]}', mime="application/json")
    )
    gw = MCPGateway(session)

    assert await gw.read_resource("data://x") == {"a": 1, "b": [2, 3]}


@sync
async def test_read_resource_falls_back_to_text_on_bad_json():
    session = fake_session(
        read_return=text_contents("not valid json {", mime="application/json")
    )
    gw = MCPGateway(session)

    assert await gw.read_resource("data://x") == "not valid json {"


@sync
async def test_read_resource_returns_plain_text_for_non_json():
    session = fake_session(read_return=text_contents("hello there", mime="text/plain"))
    gw = MCPGateway(session)

    assert await gw.read_resource("data://x") == "hello there"


@sync
async def test_read_resource_stringifies_non_text_content():
    blob = blob_contents()
    session = fake_session(read_return=blob)
    gw = MCPGateway(session)

    result = await gw.read_resource("data://x")
    assert result == str(blob.contents[0])


# Thin unwrapping passthrough


@sync
async def test_get_prompt_unwraps_messages():
    session = fake_session(messages=["m1", "m2"])
    gw = MCPGateway(session)

    result = await gw.get_prompt("cite", {"dataset": "x"})

    assert result == ["m1", "m2"]
    session.get_prompt.assert_awaited_once_with("cite", {"dataset": "x"})
