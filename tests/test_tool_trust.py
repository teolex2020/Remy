"""Tests for Self-Tooling — Tool Trust Levels (AUTON-9) — tool_trust.py."""


# ============== Unit Tests: classify_tool_source ==============


class TestClassifyToolSource:
    def test_safe_tool(self):
        from remy.core.tool_trust import TRUST_SAFE, classify_tool_source

        source = """
TOOL_NAME = "my_tool"
TOOL_DESCRIPTION = "A safe tool"
TOOL_PARAMETERS = {}

def execute(args):
    return {"result": args.get("query", "")}

def test_basic():
    assert execute({"query": "hi"}) == {"result": "hi"}
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_SAFE
        assert not result.has_network
        assert not result.has_file_write
        assert not result.has_subprocess

    def test_network_tool_is_dangerous(self):
        from remy.core.tool_trust import TRUST_DANGEROUS, classify_tool_source

        source = """
import httpx

def execute(args):
    r = httpx.get(args["url"])
    return {"status": r.status_code}
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_DANGEROUS
        assert result.has_network

    def test_subprocess_tool_is_dangerous(self):
        from remy.core.tool_trust import TRUST_DANGEROUS, classify_tool_source

        source = """
import subprocess

def execute(args):
    result = subprocess.run(["ls"], capture_output=True)
    return {"output": result.stdout}
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_DANGEROUS
        assert result.has_subprocess

    def test_eval_is_dangerous(self):
        from remy.core.tool_trust import TRUST_DANGEROUS, classify_tool_source

        source = """
def execute(args):
    return eval(args["expr"])
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_DANGEROUS
        assert result.has_dynamic_exec

    def test_file_write_is_moderate(self):
        from remy.core.tool_trust import TRUST_MODERATE, classify_tool_source

        source = """
def execute(args):
    with open("output.txt", "w") as f:
        f.write(args["data"])
    return {"status": "ok"}
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_MODERATE
        assert result.has_file_write

    def test_file_read_is_safe(self):
        from remy.core.tool_trust import TRUST_SAFE, classify_tool_source

        source = """
def execute(args):
    with open("input.txt", "r") as f:
        return {"data": f.read()}
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_SAFE
        assert not result.has_file_write

    def test_requests_module_is_dangerous(self):
        from remy.core.tool_trust import TRUST_DANGEROUS, classify_tool_source

        source = """
import requests

def execute(args):
    return requests.get(args["url"]).text
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_DANGEROUS
        assert result.has_network

    def test_from_import_detected(self):
        from remy.core.tool_trust import TRUST_DANGEROUS, classify_tool_source

        source = """
from urllib.request import urlopen

def execute(args):
    return urlopen(args["url"]).read()
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_DANGEROUS
        assert result.has_network

    def test_syntax_error_is_dangerous(self):
        from remy.core.tool_trust import TRUST_DANGEROUS, classify_tool_source

        result = classify_tool_source("def foo(:\n  pass")
        assert result.trust_level == TRUST_DANGEROUS

    def test_file_write_keyword_mode(self):
        from remy.core.tool_trust import TRUST_MODERATE, classify_tool_source

        source = """
def execute(args):
    with open("output.txt", mode="w") as f:
        f.write("data")
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_MODERATE
        assert result.has_file_write

    def test_multiple_risks(self):
        from remy.core.tool_trust import TRUST_DANGEROUS, classify_tool_source

        source = """
import httpx
import subprocess

def execute(args):
    r = httpx.get(args["url"])
    subprocess.run(["echo", r.text])
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_DANGEROUS
        assert result.has_network
        assert result.has_subprocess
        assert len(result.reasons) >= 2


# ============== Unit Tests: TrustClassification ==============


class TestTrustClassification:
    def test_defaults(self):
        from remy.core.tool_trust import TRUST_SAFE, TrustClassification

        tc = TrustClassification(trust_level=TRUST_SAFE)
        assert tc.trust_level == TRUST_SAFE
        assert tc.has_network is False
        assert tc.reasons == []


# ============== Unit Tests: should_auto_approve ==============


class TestShouldAutoApprove:
    def test_safe_tool_approved(self, tmp_path):
        from remy.core.tool_trust import should_auto_approve

        tool_file = tmp_path / "safe_tool.py"
        tool_file.write_text("""
def execute(args):
    return {"result": "ok"}
""")
        approved, reason = should_auto_approve(tool_file)
        assert approved is True
        assert "Safe" in reason

    def test_dangerous_tool_denied(self, tmp_path):
        from remy.core.tool_trust import should_auto_approve

        tool_file = tmp_path / "danger_tool.py"
        tool_file.write_text("""
import subprocess
def execute(args):
    subprocess.run(args["cmd"])
""")
        approved, reason = should_auto_approve(tool_file)
        assert approved is False
        assert "human approval" in reason.lower() or "Dangerous" in reason

    def test_moderate_tool_approved(self, tmp_path):
        from remy.core.tool_trust import should_auto_approve

        tool_file = tmp_path / "write_tool.py"
        tool_file.write_text("""
def execute(args):
    with open("out.txt", "w") as f:
        f.write("data")
""")
        approved, reason = should_auto_approve(tool_file)
        assert approved is True
        assert "Moderate" in reason

    def test_missing_file(self):
        from remy.core.tool_trust import should_auto_approve

        approved, reason = should_auto_approve("/nonexistent/path.py")
        assert approved is False


# ============== Unit Tests: find_retired_tools ==============


class TestFindRetiredTools:
    def test_no_retired(self):
        import time

        from remy.core.tool_trust import find_retired_tools

        tools = [
            {
                "name": "tool1",
                "status": "approved",
                "last_used": time.time(),
                "created_at": time.time(),
            },
        ]
        assert find_retired_tools(tools) == []

    def test_old_tool_retired(self):
        import time

        from remy.core.tool_trust import find_retired_tools

        old_time = time.time() - (40 * 86400)  # 40 days ago
        tools = [
            {
                "name": "old_tool",
                "status": "approved",
                "last_used": old_time,
                "created_at": old_time,
            },
        ]
        retired = find_retired_tools(tools)
        assert "old_tool" in retired

    def test_draft_tool_not_retired(self):
        import time

        from remy.core.tool_trust import find_retired_tools

        old_time = time.time() - (40 * 86400)
        tools = [
            {
                "name": "draft_tool",
                "status": "draft",
                "last_used": old_time,
                "created_at": old_time,
            },
        ]
        assert find_retired_tools(tools) == []

    def test_recently_used_not_retired(self):
        import time

        from remy.core.tool_trust import find_retired_tools

        tools = [
            {
                "name": "active_tool",
                "status": "approved",
                "last_used": time.time() - 86400,  # 1 day ago
                "created_at": time.time() - (60 * 86400),
            },
        ]
        assert find_retired_tools(tools) == []


# ============== Unit Tests: format_trust_report ==============


class TestFormatTrustReport:
    def test_safe_report(self):
        from remy.core.tool_trust import TRUST_SAFE, TrustClassification, format_trust_report

        tc = TrustClassification(trust_level=TRUST_SAFE)
        text = format_trust_report(tc)
        assert "GREEN" in text
        assert "safe" in text

    def test_dangerous_report(self):
        from remy.core.tool_trust import TRUST_DANGEROUS, TrustClassification, format_trust_report

        tc = TrustClassification(
            trust_level=TRUST_DANGEROUS,
            has_network=True,
            has_subprocess=True,
            reasons=["Network module: httpx", "Subprocess call"],
        )
        text = format_trust_report(tc)
        assert "RED" in text
        assert "Network" in text
        assert "Subprocess" in text


# ============== Integration: AST patterns ==============


class TestASTPatterns:
    def test_os_system_detected(self):
        from remy.core.tool_trust import TRUST_DANGEROUS, classify_tool_source

        source = """
import os
def execute(args):
    os.system(args["cmd"])
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_DANGEROUS

    def test_socket_detected(self):
        from remy.core.tool_trust import classify_tool_source

        source = """
import socket
def execute(args):
    s = socket.socket()
"""
        result = classify_tool_source(source)
        assert result.has_network is True

    def test_math_only_is_safe(self):
        from remy.core.tool_trust import TRUST_SAFE, classify_tool_source

        source = """
import math
def execute(args):
    return {"result": math.sqrt(args["n"])}
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_SAFE

    def test_json_is_safe(self):
        from remy.core.tool_trust import TRUST_SAFE, classify_tool_source

        source = """
import json
def execute(args):
    return json.loads(args["text"])
"""
        result = classify_tool_source(source)
        assert result.trust_level == TRUST_SAFE
