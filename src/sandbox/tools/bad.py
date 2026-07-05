TOOL_NAME = "bad"
TOOL_DESCRIPTION = "Fails tests"
TOOL_PARAMETERS = {}
TOOL_REQUIRED = []


def execute() -> str:
    return "ok"


def test_execute():
    assert 1 == 2
