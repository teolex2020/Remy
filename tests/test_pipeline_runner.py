from remy.core import pipeline_runner
from types import SimpleNamespace
import sys
import pytest


def test_web_search_ranking_filters_noise_for_currency_query():
    results = [
        {
            "title": "курс доллара translation",
            "body": "Russian-English translation examples",
            "href": "https://context.reverso.net/translation/russian-english/курс+доллара",
        },
        {
            "title": "Курс Доллара",
            "body": "Song on Apple Music",
            "href": "https://music.apple.com/gb/song/курс-доллара/1082241833",
        },
        {
            "title": "USD exchange rate in Ukrainian banks",
            "body": "Cash currency exchange rates in banks of Ukraine",
            "href": "https://finance.liga.net/en/currency/card/currency/usd/currencyb/filter",
        },
    ]

    ranked = pipeline_runner._rank_search_results("курс доллара", results, limit=3)

    assert [r["href"] for r in ranked] == [
        "https://finance.liga.net/en/currency/card/currency/usd/currencyb/filter"
    ]


def test_web_search_ranking_allows_translation_when_query_asks_for_translation():
    results = [
        {
            "title": "курс доллара translation",
            "body": "Russian-English translation examples",
            "href": "https://context.reverso.net/translation/russian-english/курс+доллара",
        },
    ]

    ranked = pipeline_runner._rank_search_results("translate курс доллара", results, limit=1)

    assert ranked == results


def test_page_scrape_text_normalizer_collapses_large_gaps():
    raw = "  Product title  \n\n\n\n   Price:    1200 UAH   \n\n\nSpecs   here  "

    assert pipeline_runner._normalize_scraped_text(raw) == "Product title\n\nPrice: 1200 UAH\n\nSpecs here"


@pytest.mark.asyncio
async def test_memory_search_skip_empty_result(monkeypatch):
    class _Brain:
        def search(self, *_args, **_kwargs):
            return []

    import remy.core.agent_tools as agent_tools

    monkeypatch.setattr(agent_tools, "brain", _Brain())

    output = await pipeline_runner.run_single_step(
        {"type": "memory_search", "config": {"query": "missing", "skip_empty_result": True}},
        "",
    )

    assert output == ""


@pytest.mark.asyncio
async def test_memory_save_dedup_guard_skips_existing_memory(monkeypatch):
    class _Brain:
        def __init__(self):
            self.saved = []

        def search(self, *_args, **_kwargs):
            return [SimpleNamespace(content="Remember this")]

        def store(self, content, tags=None, metadata=None):
            self.saved.append({"content": content, "tags": tags or [], "metadata": metadata or {}})

    import remy.core.agent_tools as agent_tools

    fake_brain = _Brain()
    monkeypatch.setattr(agent_tools, "brain", fake_brain)

    output = await pipeline_runner.run_single_step(
        {"type": "memory_save", "config": {"text": " remember   this ", "dedup_guard": True}},
        "",
    )

    assert output == "Skipped duplicate memory save"
    assert fake_brain.saved == []


@pytest.mark.asyncio
async def test_memory_save_uses_input_source_when_text_is_missing(monkeypatch):
    class _Brain:
        def __init__(self):
            self.saved = []

        def store(self, content, tags=None, metadata=None):
            self.saved.append({"content": content, "tags": tags or [], "metadata": metadata or {}})
            return SimpleNamespace(id="rec-1")

    import remy.core.agent_tools as agent_tools

    fake_brain = _Brain()
    monkeypatch.setattr(agent_tools, "brain", fake_brain)

    output = await pipeline_runner.run_single_step(
        {"type": "memory_save", "config": {"input_source": "source text", "tags": "source"}},
        "",
    )

    assert output == "Saved to memory (11 characters)"
    assert fake_brain.saved == [{
        "content": "source text",
        "tags": ["source"],
        "metadata": {"source": "pipeline"},
    }]


@pytest.mark.asyncio
async def test_http_request_uses_local_secret_for_authorization(monkeypatch):
    captured = {}

    class _Response:
        text = "ok"

        def raise_for_status(self):
            return None

    class _HttpClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = headers or {}
            return _Response()

    from remy.config.settings import settings

    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "sk-secret-value")
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(AsyncClient=_HttpClient))

    output = await pipeline_runner.run_single_step(
        {
            "type": "http_request",
            "config": {
                "url": "https://api.example.test",
                "auth_secret_key": "openrouter_api_key",
                "auth_scheme": "Bearer",
            },
        },
        "",
    )

    assert output == "ok"
    assert captured["headers"]["Authorization"] == "Bearer sk-secret-value"


@pytest.mark.asyncio
async def test_http_request_error_is_user_friendly(monkeypatch):
    class _HttpError(Exception):
        response = SimpleNamespace(status_code=401)

    class _Response:
        text = ""

        def raise_for_status(self):
            raise _HttpError("401 Unauthorized")

    class _HttpClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(AsyncClient=_HttpClient))

    output = await pipeline_runner.run_single_step(
        {"type": "http_request", "config": {"url": "https://api.example.test/private"}},
        "",
    )

    assert output.startswith("[HTTP error:")
    assert "Authorization secret" in output
    assert "Try Test Connection" in output


@pytest.mark.asyncio
async def test_http_request_rejects_unknown_method_before_network():
    output = await pipeline_runner.run_single_step(
        {"type": "http_request", "config": {"url": "https://api.example.test", "method": "DELETE"}},
        "",
    )

    assert output == "[HTTP error: method must be GET or POST]"


@pytest.mark.asyncio
async def test_search_blocks_clamp_result_limits(monkeypatch):
    captured = {}

    class _DDGS:
        def text(self, query, max_results=5):
            captured["web_query"] = query
            captured["web_max_results"] = max_results
            return [
                {"title": f"Result {index}", "body": "topic snippet", "href": f"https://example.test/{index}"}
                for index in range(max_results)
            ]

    class _Brain:
        def search(self, query, top_k=5):
            captured["memory_query"] = query
            captured["memory_top_k"] = top_k
            return [SimpleNamespace(content=f"Memory {index}") for index in range(top_k)]

    import remy.core.agent_tools as agent_tools

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=lambda: _DDGS()))
    monkeypatch.setattr(agent_tools, "brain", _Brain())

    web_output = await pipeline_runner.run_single_step(
        {"type": "web_search", "config": {"query": "topic", "num_results": 99, "fetch_content": False}},
        "",
    )
    memory_output = await pipeline_runner.run_single_step(
        {"type": "memory_search", "config": {"query": "topic", "limit": 99}},
        "",
    )

    assert "WEB SEARCH RESULTS" in web_output
    assert "Memory 19" in memory_output
    assert captured["web_max_results"] == 20
    assert captured["memory_top_k"] == 20


@pytest.mark.asyncio
async def test_page_scrape_error_is_user_friendly_for_unreadable_page(monkeypatch):
    class _Response:
        text = "<html><script>window.app = true</script></html>"

        def raise_for_status(self):
            return None

    class _HttpClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(AsyncClient=_HttpClient))

    output = await pipeline_runner.run_single_step(
        {"type": "page_scrape", "config": {"url": "https://example.test/app", "mode": "text"}},
        "",
    )

    assert output.startswith("[Scrape error:")
    assert "Page could not be read as clean text" in output
    assert "HTTP Request" in output


@pytest.mark.asyncio
async def test_all_pipeline_block_runners_smoke(tmp_path, monkeypatch):
    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class _Brain:
        def __init__(self):
            self.saved = []

        def search(self, query, top_k=5, **_kwargs):
            return [SimpleNamespace(content=f"memory result for {query}")]

        def store(self, content, tags=None, metadata=None):
            self.saved.append({"content": content, "tags": tags or [], "metadata": metadata or {}})
            return SimpleNamespace(id="rec-1")

    class _Response:
        text = "http ok"

        def raise_for_status(self):
            return None

    class _HttpClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, *_args, **_kwargs):
            return _Response()

        async def post(self, *_args, **_kwargs):
            return _Response()

    class _DDGS:
        def text(self, *_args, **_kwargs):
            return [{"title": "Result", "body": "Snippet", "href": "https://example.test"}]

    fake_brain = _Brain()
    import remy.core.agent_tools as agent_tools
    import remy.core.llm as llm_module
    from remy.config.settings import settings

    monkeypatch.setattr(agent_tools, "brain", fake_brain)
    monkeypatch.setattr(agent_tools, "brain_lock", _Lock())
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        llm_module,
        "get_llm",
        lambda *_args, **_kwargs: SimpleNamespace(invoke=lambda _prompt: SimpleNamespace(content="llm ok")),
    )
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(AsyncClient=_HttpClient))
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=lambda: _DDGS()))

    assert await pipeline_runner.run_single_step({"type": "llm_call", "config": {"prompt": "hello"}}, "") == "llm ok"
    assert "WEB SEARCH RESULTS" in await pipeline_runner.run_single_step(
        {"type": "web_search", "config": {"query": "topic", "num_results": 1, "fetch_content": False}},
        "",
    )
    assert "memory result for topic" in await pipeline_runner.run_single_step(
        {"type": "memory_search", "config": {"query": "topic"}},
        "",
    )
    assert await pipeline_runner.run_single_step(
        {"type": "memory_save", "config": {"text": "remember this", "tags": "a,b"}},
        "",
    ) == "Saved to memory (13 characters)"
    assert fake_brain.saved[0]["tags"] == ["a", "b"]
    assert await pipeline_runner.run_single_step({"type": "http_request", "config": {"url": "https://example.test"}}, "") == "http ok"
    scraped = await pipeline_runner.run_single_step({"type": "page_scrape", "config": {"url": "https://example.test", "mode": "text"}}, "")
    assert "PAGE SCRAPE" in scraped and "http ok" in scraped
    assert await pipeline_runner.run_single_step({"type": "template", "config": {"text": "plain"}}, "") == "plain"
    assert await pipeline_runner.run_single_step(
        {"type": "merge", "config": {"_merge_inputs": ["a", "b"], "mode": "combine_text", "separator": " + "}},
        "",
    ) == "a + b"
    assert await pipeline_runner.run_single_step({"type": "delay", "config": {"seconds": 0}}, "input") == "input"
    assert await pipeline_runner.run_single_step(
        {"type": "filter", "config": {"_data_ref": "allow me", "operator": "contains", "value": "allow"}},
        "",
    ) == "allow me"
    assert await pipeline_runner.run_single_step({"type": "set_variable", "config": {"value": "42"}}, "") == "42"
    assert await pipeline_runner.run_single_step(
        {"type": "parse_json", "config": {"text": '{"name":"Remy"}', "path": "$.name"}},
        "",
    ) == "Remy"
    assert await pipeline_runner.run_single_step(
        {"type": "transform", "config": {"text": "  Remy  ", "mode": "trim"}},
        "",
    ) == "Remy"
    assert await pipeline_runner.run_single_step(
        {"type": "notification", "config": {"title": "Done", "message": "Ok"}},
        "",
    ) == "Done: Ok"
    assert await pipeline_runner.run_single_step(
        {"type": "file_write", "config": {"filename": "note.txt", "text": "file ok"}},
        "",
    ) == "Saved file: note.txt (7 characters)"
    assert await pipeline_runner.run_single_step(
        {"type": "file_read", "config": {"filename": "note.txt"}},
        "",
    ) == "file ok"
    assert await pipeline_runner.run_single_step(
        {"type": "code", "config": {"mode": "safe_expression", "code": "input.strip().upper()"}},
        " remy ",
    ) == "REMY"
    assert await pipeline_runner.run_single_step(
        {"type": "error_handler", "config": {"fallback_text": "fallback {{error}}"}},
        "[boom]",
    ) == "fallback {{error}}"
    assert await pipeline_runner.run_single_step(
        {"type": "router", "config": {"_data_ref": "billing", "routes": [{"operator": "contains", "value": "billing"}]}},
        "",
    ) == "1"


@pytest.mark.asyncio
async def test_code_block_blocks_local_script_without_explicit_opt_in():
    output = await pipeline_runner.run_single_step(
        {
            "type": "code",
            "config": {
                "mode": "local_script",
                "language": "python",
                "code": "result = input.upper()",
                "allow_local_execution": False,
            },
        },
        "hello",
    )

    assert output == "[Code blocked: enable local execution in this block to run scripts on this computer.]"


@pytest.mark.asyncio
async def test_code_block_runs_local_python_when_enabled(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    output = await pipeline_runner.run_single_step(
        {
            "type": "code",
            "config": {
                "mode": "local_script",
                "language": "python",
                "code": "result = input.upper()",
                "allow_local_execution": True,
                "timeout_seconds": 3,
                "max_output_chars": 100,
            },
        },
        "hello",
    )

    assert output == "HELLO"


def test_router_structured_condition_operators():
    assert pipeline_runner._route_condition_simple_match(
        {"operator": "contains", "value": "invoice"}, "new invoice arrived"
    ) is True
    assert pipeline_runner._route_condition_simple_match(
        {"operator": "not_contains", "value": "spam"}, "new invoice arrived"
    ) is True
    assert pipeline_runner._route_condition_simple_match(
        {"operator": "equals", "value": "urgent"}, "urgent"
    ) is True
    assert pipeline_runner._route_condition_simple_match(
        {"operator": "starts_with", "value": "hello"}, "hello world"
    ) is True
    assert pipeline_runner._route_condition_simple_match(
        {"operator": "fallback", "value": ""}, "anything"
    ) is None


@pytest.mark.asyncio
async def test_pipeline_run_events_include_step_identity():
    steps = [{"id": "s42", "type": "template", "label": "Custom step", "config": {"text": "ok"}}]

    events = [event async for event in pipeline_runner.run_pipeline_steps(steps, "hello")]

    assert events[1]["type"] == "step_start"
    assert events[1]["id"] == "s42"
    assert events[1]["step_type"] == "template"
    assert events[2]["type"] == "step_done"
    assert events[2]["id"] == "s42"
    assert events[2]["step_type"] == "template"


@pytest.mark.asyncio
async def test_pipeline_unknown_block_is_reported_as_error_output():
    steps = [{"id": "s42", "type": "unknown_block", "label": "Custom step", "config": {}}]

    events = [event async for event in pipeline_runner.run_pipeline_steps(steps, "hello")]

    assert events[2]["type"] == "step_error"
    assert events[2]["id"] == "s42"
    assert events[2]["step_type"] == "unknown_block"
    assert events[2]["error"] == "[Unknown block type: unknown_block]"


@pytest.mark.asyncio
async def test_pipeline_step_can_use_pinned_output_without_runner():
    step = {
        "id": "s1",
        "type": "unknown_block",
        "label": "Pinned",
        "config": {"_pinned_enabled": True, "_pinned_output": "cached result"},
    }

    events = [event async for event in pipeline_runner.run_pipeline_steps([step], "hello")]

    assert events[2]["type"] == "step_done"
    assert events[2]["output"] == "cached result"
    assert events[-1]["output"] == "cached result"


@pytest.mark.asyncio
async def test_pipeline_router_follows_selected_output_branch():
    steps = [
        {
            "id": "s1",
            "type": "router",
            "label": "Router",
            "config": {
                "_data_ref": "{{input}}",
                "mode": "first_match",
                "routes": [
                    {"label": "General", "operator": "contains", "value": "general"},
                    {"label": "Billing", "operator": "contains", "value": "billing"},
                ],
            },
            "_df_id": 1,
            "_connections": {
                "output_1": {"connections": [{"node": "2"}]},
                "output_2": {"connections": [{"node": "3"}]},
            },
        },
        {"id": "s2", "type": "template", "label": "Wrong", "config": {"text": "wrong"}, "_df_id": 2, "_connections": {}},
        {"id": "s3", "type": "template", "label": "Right", "config": {"text": "right"}, "_df_id": 3, "_connections": {}},
    ]

    events = [event async for event in pipeline_runner.run_pipeline_steps(steps, "billing question")]

    done = [event for event in events if event["type"] == "step_done"]
    assert [event["id"] for event in done] == ["s1", "s3"]
    assert done[0]["output"] == "Selected routes: output_2"
    assert events[-1]["output"] == "right"


@pytest.mark.asyncio
async def test_pipeline_router_can_fan_out_to_multiple_matching_branches():
    steps = [
        {
            "id": "s1",
            "type": "router",
            "label": "Router",
            "config": {
                "_data_ref": "{{input}}",
                "mode": "all_matching",
                "routes": [
                    {"label": "Support", "operator": "contains", "value": "support"},
                    {"label": "Billing", "operator": "contains", "value": "billing"},
                ],
            },
            "_df_id": 1,
            "_connections": {
                "output_1": {"connections": [{"node": "2"}]},
                "output_2": {"connections": [{"node": "3"}]},
            },
        },
        {"id": "s2", "type": "template", "label": "Support", "config": {"text": "support branch"}, "_df_id": 2, "_connections": {}},
        {"id": "s3", "type": "template", "label": "Billing", "config": {"text": "billing branch"}, "_df_id": 3, "_connections": {}},
    ]

    events = [event async for event in pipeline_runner.run_pipeline_steps(steps, "support and billing")]

    done = [event for event in events if event["type"] == "step_done"]
    assert [event["id"] for event in done] == ["s1", "s2", "s3"]
    assert done[0]["output"] == "Selected routes: output_1, output_2"
    assert events[-1]["output"] == "support branch\n\n---\n\nbilling branch"


@pytest.mark.asyncio
async def test_pipeline_merge_waits_for_router_branches_before_continuing():
    steps = [
        {
            "id": "s1",
            "type": "router",
            "label": "Router",
            "config": {
                "_data_ref": "{{input}}",
                "mode": "all",
                "routes": [
                    {"label": "A", "operator": "always", "value": ""},
                    {"label": "B", "operator": "always", "value": ""},
                ],
            },
            "_df_id": 1,
            "_connections": {
                "output_1": {"connections": [{"node": "2", "output": "input_1"}]},
                "output_2": {"connections": [{"node": "3", "output": "input_1"}]},
            },
        },
        {
            "id": "s2",
            "type": "template",
            "label": "A",
            "config": {"text": "branch A"},
            "_df_id": 2,
            "_connections": {"output_1": {"connections": [{"node": "4", "output": "input_1"}]}},
        },
        {
            "id": "s3",
            "type": "template",
            "label": "B",
            "config": {"text": "branch B"},
            "_df_id": 3,
            "_connections": {"output_1": {"connections": [{"node": "4", "output": "input_2"}]}},
        },
        {
            "id": "s4",
            "type": "merge",
            "label": "Merge",
            "config": {"mode": "combine_text", "separator": " | ", "input_count": 2},
            "_df_id": 4,
            "_inputs": {
                "input_1": {"connections": [{"node": "2", "input": "output_1"}]},
                "input_2": {"connections": [{"node": "3", "input": "output_1"}]},
            },
            "_connections": {"output_1": {"connections": [{"node": "5", "output": "input_1"}]}},
        },
        {
            "id": "s5",
            "type": "template",
            "label": "Final",
            "config": {"text": "Final: {{prev}}"},
            "_df_id": 5,
            "_connections": {},
        },
    ]

    events = [event async for event in pipeline_runner.run_pipeline_steps(steps, "hello")]

    done = [event for event in events if event["type"] == "step_done"]
    assert [event["id"] for event in done] == ["s1", "s2", "s3", "s4", "s5"]
    assert done[3]["output"] == "branch A | branch B"
    assert events[-1]["output"] == "Final: branch A | branch B"


@pytest.mark.asyncio
async def test_pipeline_utility_blocks_parse_transform_and_variables():
    steps = [
        {"id": "s1", "type": "parse_json", "label": "Parse", "config": {"text": '{"items":[{"title":" Hello " }]}', "path": "$.items[0].title"}},
        {"id": "s2", "type": "transform", "label": "Trim", "config": {"text": "{{prev}}", "mode": "trim"}},
        {"id": "s3", "type": "set_variable", "label": "Set", "config": {"name": "title", "value": "{{prev}}"}},
        {"id": "s4", "type": "template", "label": "Use", "config": {"text": "Title={{title}}"}},
    ]

    events = [event async for event in pipeline_runner.run_pipeline_steps(steps, "")]

    assert events[-1]["output"] == "Title=Hello"


@pytest.mark.asyncio
async def test_pipeline_filter_can_stop_branch():
    steps = [
        {"id": "s1", "type": "filter", "label": "Filter", "config": {"_data_ref": "{{input}}", "operator": "contains", "value": "allow"}},
        {"id": "s2", "type": "template", "label": "After", "config": {"text": "should not run"}},
    ]

    events = [event async for event in pipeline_runner.run_pipeline_steps(steps, "deny")]

    done = [event for event in events if event["type"] == "step_done"]
    assert [event["id"] for event in done] == ["s1"]
    assert done[0]["output"] == "Stopped by filter"
    assert events[-1]["output"] == "deny"


@pytest.mark.asyncio
async def test_pipeline_error_output_routes_to_error_handler():
    steps = [
        {
            "id": "s1",
            "type": "parse_json",
            "label": "Parse",
            "config": {"text": "not json", "path": "$"},
            "_df_id": 1,
            "_connections": {
                "output_1": {"connections": [{"node": "3", "output": "input_1"}]},
                "output_2": {"connections": [{"node": "2", "output": "input_1"}]},
            },
        },
        {
            "id": "s2",
            "type": "error_handler",
            "label": "Recover",
            "config": {"fallback_text": "fallback: {{error}}"},
            "_df_id": 2,
            "_connections": {"output_1": {"connections": [{"node": "3", "output": "input_1"}]}},
        },
        {
            "id": "s3",
            "type": "template",
            "label": "Final",
            "config": {"text": "Final {{prev}}"},
            "_df_id": 3,
            "_connections": {},
        },
    ]

    events = [event async for event in pipeline_runner.run_pipeline_steps(steps, "")]

    assert [event["type"] for event in events if event["type"].startswith("step_")] == [
        "step_start",
        "step_error",
        "step_start",
        "step_done",
        "step_start",
        "step_done",
    ]
    done = [event for event in events if event["type"] == "step_done"]
    assert [event["id"] for event in done] == ["s2", "s3"]
    assert events[-1]["output"].startswith("Final fallback: [JSON parse error:")


@pytest.mark.asyncio
async def test_pipeline_file_write_and_read_use_workflow_files_dir(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    steps = [
        {"id": "s1", "type": "file_write", "label": "Write", "config": {"filename": "note.txt", "text": "hello", "mode": "overwrite"}},
        {"id": "s2", "type": "file_read", "label": "Read", "config": {"filename": "note.txt", "max_chars": 100}},
    ]

    events = [event async for event in pipeline_runner.run_pipeline_steps(steps, "")]

    assert events[-1]["output"] == "hello"
    assert (tmp_path / "workflow_files" / "note.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_pipeline_file_write_overwrite_uses_atomic_write(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.file_utils import atomic_write as real_atomic_write

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    calls = []

    def _spy_atomic_write(path, content, encoding="utf-8"):
        calls.append((path, content))
        return real_atomic_write(path, content, encoding)

    monkeypatch.setattr(pipeline_runner, "atomic_write", _spy_atomic_write)

    result = await pipeline_runner.run_single_step(
        {"type": "file_write", "config": {"filename": "note.txt", "text": "atomic file", "mode": "overwrite"}},
        "",
    )

    assert result == "Saved file: note.txt (11 characters)"
    assert len(calls) == 1
    assert str(calls[0][0]).endswith("note.txt")
    assert calls[0][1] == "atomic file"


@pytest.mark.asyncio
async def test_pipeline_file_write_rejects_unknown_mode(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    result = await pipeline_runner.run_single_step(
        {"type": "file_write", "config": {"filename": "note.txt", "text": "unsafe", "mode": "replace"}},
        "",
    )

    assert result == "[File write error: mode must be overwrite or append]"
    assert not (tmp_path / "workflow_files" / "note.txt").exists()


@pytest.mark.asyncio
async def test_pipeline_notification_stores_local_notification(monkeypatch):
    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class _Brain:
        def __init__(self):
            self.records = []

        def store(self, content, tags=None, metadata=None):
            self.records.append({
                "content": content,
                "tags": tags or [],
                "metadata": metadata or {},
            })

    fake_brain = _Brain()
    import remy.core.agent_tools as agent_tools

    monkeypatch.setattr(agent_tools, "brain", fake_brain)
    monkeypatch.setattr(agent_tools, "brain_lock", _Lock())

    events = [event async for event in pipeline_runner.run_pipeline_steps([
        {"id": "s1", "type": "notification", "label": "Notify", "config": {"title": "Done", "message": "Workflow finished"}},
    ], "")]

    assert events[-1]["output"] == "Done: Workflow finished"
    assert fake_brain.records == [{
        "content": "Done: Workflow finished",
        "tags": ["workflow-notification", "notification"],
        "metadata": {"source": "workflow", "title": "Done"},
    }]
