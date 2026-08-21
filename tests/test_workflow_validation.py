from remy.core.workflow_validation import validate_visual_workflow_graph, validate_workflow_step_configs


def _flow(data):
    return {"drawflow": {"Home": {"data": data}}}


def test_visual_workflow_validation_accepts_connected_router_merge_graph():
    errors = validate_visual_workflow_graph(
        workflow_label="Pipeline",
        entry_name="start",
        terminal_name="result",
        steps=[
            {"id": "s2", "type": "router", "_df_id": 2},
            {"id": "s3", "type": "template", "_df_id": 3},
            {"id": "s4", "type": "template", "_df_id": 4},
            {"id": "s5", "type": "merge", "_df_id": 5},
        ],
        drawflow_data=_flow({
            "1": {"name": "start", "outputs": {"output_1": {"connections": [{"node": "2"}]}}},
            "2": {"name": "router", "outputs": {
                "output_1": {"connections": [{"node": "3"}]},
                "output_2": {"connections": [{"node": "4"}]},
            }},
            "3": {"name": "template", "outputs": {"output_1": {"connections": [{"node": "5"}]}}},
            "4": {"name": "template", "outputs": {"output_1": {"connections": [{"node": "5"}]}}},
            "5": {
                "name": "merge",
                "inputs": {
                    "input_1": {"connections": [{"node": "3"}]},
                    "input_2": {"connections": [{"node": "4"}]},
                },
                "outputs": {"output_1": {"connections": [{"node": "6"}]}},
            },
            "6": {"name": "result", "inputs": {"input_1": {"connections": [{"node": "5"}]}}},
        }),
    )

    assert errors == []


def test_visual_workflow_validation_finds_broken_graph_edges():
    errors = validate_visual_workflow_graph(
        workflow_label="Pipeline",
        entry_name="start",
        terminal_name="result",
        steps=[
            {"id": "s2", "type": "router", "_df_id": 2},
            {"id": "s3", "type": "merge", "_df_id": 3},
            {"id": "s4", "type": "template", "_df_id": 4},
        ],
        drawflow_data=_flow({
            "1": {"name": "start", "outputs": {"output_1": {"connections": [{"node": "2"}]}}},
            "2": {"name": "router", "outputs": {
                "output_1": {"connections": [{"node": "3"}]},
                "output_2": {"connections": []},
            }},
            "3": {
                "name": "merge",
                "inputs": {"input_1": {"connections": [{"node": "2"}]}},
                "outputs": {},
            },
            "4": {"name": "template", "outputs": {"output_1": {"connections": [{"node": "5"}]}}},
            "5": {"name": "result", "inputs": {"input_1": {"connections": [{"node": "4"}]}}},
        }),
    )

    assert "Pipeline has no connected path from start to result." in errors
    assert "Pipeline block 4 (template) is not reachable from start." in errors
    assert "Pipeline Router block 2 has an unconnected output_2." in errors
    assert "Pipeline Merge block 3 needs at least two connected inputs." in errors
    assert "Pipeline block 3 (merge) has no outgoing connection." in errors


def test_workflow_step_config_validation_catches_invalid_utility_blocks():
    errors = validate_workflow_step_configs(
        workflow_label="Pipeline",
        steps=[
            {"id": "s1", "type": "delay", "label": "Delay", "config": {"seconds": 301}},
            {"id": "s2", "type": "filter", "label": "Filter", "config": {"operator": "contains", "value": ""}},
            {"id": "s3", "type": "set_variable", "label": "Var", "config": {"name": "1bad", "value": "x"}},
            {"id": "s4", "type": "parse_json", "label": "JSON", "config": {"text": "", "path": ""}},
            {"id": "s5", "type": "transform", "label": "Tx", "config": {"mode": "extract_regex", "pattern": ""}},
            {"id": "s6", "type": "file_write", "label": "File", "config": {"filename": "../bad.txt", "mode": "replace"}},
            {"id": "s7", "type": "code", "label": "Code", "config": {"mode": "bad", "language": "ruby", "code": "", "timeout_seconds": 31}},
            {"id": "s8", "type": "http_request", "label": "HTTP", "config": {"method": "DELETE"}},
            {"id": "s9", "type": "web_search", "label": "Web", "config": {"num_results": 99}},
            {"id": "s10", "type": "memory_search", "label": "Memory", "config": {"limit": 99}},
            {"id": "s11", "type": "memory_save", "label": "Save", "config": {"tags": "empty"}},
        ],
    )

    assert "Pipeline step 1 (Delay) Delay must be between 0 and 300 seconds." in errors
    assert "Pipeline step 2 (Filter) Filter value is required for this operator." in errors
    assert "Pipeline step 3 (Var) variable name must use letters, numbers, and underscores, and cannot start with a number." in errors
    assert "Pipeline step 4 (JSON) JSON text is required." in errors
    assert "Pipeline step 4 (JSON) JSON path is required." in errors
    assert "Pipeline step 5 (Tx) regex pattern is required." in errors
    assert "Pipeline step 6 (File) file name must be a simple file name inside workflow_files." in errors
    assert "Pipeline step 6 (File) File Write text is required." in errors
    assert "Pipeline step 6 (File) File Write mode must be overwrite or append." in errors
    assert "Pipeline step 7 (Code) code mode is not supported." in errors
    assert "Pipeline step 7 (Code) code language is not supported." in errors
    assert "Pipeline step 7 (Code) Code / Script block cannot be empty." in errors
    assert "Pipeline step 7 (Code) code timeout must be between 0 and 30 seconds." in errors
    assert "Pipeline step 8 (HTTP) HTTP method must be GET or POST." in errors
    assert "Pipeline step 9 (Web) Web Search results must be between 1 and 10." in errors
    assert "Pipeline step 10 (Memory) Memory Search results must be between 1 and 20." in errors
    assert "Pipeline step 11 (Save) Save to Memory text is required." in errors


def test_workflow_step_config_validation_accepts_valid_utility_blocks():
    errors = validate_workflow_step_configs(
        workflow_label="Pipeline",
        steps=[
            {"id": "s1", "type": "delay", "label": "Delay", "config": {"seconds": 1}},
            {"id": "s2", "type": "filter", "label": "Filter", "config": {"operator": "contains", "value": "ok"}},
            {"id": "s3", "type": "set_variable", "label": "Var", "config": {"name": "title_1", "value": "x"}},
            {"id": "s4", "type": "parse_json", "label": "JSON", "config": {"text": "{}", "path": "$"}},
            {"id": "s5", "type": "transform", "label": "Tx", "config": {"mode": "trim"}},
            {"id": "s6", "type": "file_read", "label": "File", "config": {"filename": "notes.txt"}},
            {"id": "s7", "type": "file_write", "label": "Write", "config": {"filename": "report.txt", "text": "{{prev}}", "mode": "append"}},
            {"id": "s8", "type": "code", "label": "Code", "config": {"mode": "safe_expression", "language": "python", "code": "input.strip()"}},
            {"id": "s9", "type": "http_request", "label": "HTTP", "config": {"method": "POST"}},
            {"id": "s10", "type": "web_search", "label": "Web", "config": {"num_results": 10}},
            {"id": "s11", "type": "memory_search", "label": "Memory", "config": {"limit": 20}},
            {"id": "s12", "type": "memory_save", "label": "Save", "config": {"input_source": "{{prev}}", "tags": "ok"}},
        ],
    )

    assert errors == []


def test_workflow_step_config_validation_accepts_visible_pipeline_blocks():
    errors = validate_workflow_step_configs(
        workflow_label="Pipeline",
        steps=[
            {"id": "s1", "type": "page_scrape", "label": "Scrape", "config": {"url": "https://example.test", "mode": "text", "max_chars": 12000}},
            {"id": "s2", "type": "condition", "label": "Condition", "config": {"condition": "contains useful data"}},
            {"id": "s3", "type": "loop", "label": "Loop", "config": {"max_iterations": 5}},
        ],
    )

    assert errors == []


def test_workflow_step_config_validation_catches_bad_scraper_and_loop_config():
    errors = validate_workflow_step_configs(
        workflow_label="Pipeline",
        steps=[
            {"id": "s1", "type": "page_scrape", "label": "Scrape", "config": {"mode": "full_html", "max_chars": 100}},
            {"id": "s2", "type": "loop", "label": "Loop", "config": {"max_iterations": 25}},
        ],
    )

    assert "Pipeline step 1 (Scrape) Page Scraper extract mode must be text, title, or links." in errors
    assert "Pipeline step 1 (Scrape) Page Scraper max characters must be between 500 and 50000." in errors
    assert "Pipeline step 2 (Loop) Loop max iterations must be between 1 and 20." in errors
