"""Tests for P3.1 browser memory integration — worker hints, orchestrator routing, metrics, flow sequences."""


# ============== Browser Worker Prompt Hints ==============


class TestBrowserWorkerPromptHints:
    def test_prompt_includes_memory_hints_when_available(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            mem.record_browser_failure(
                tool="browser_act",
                action="click",
                url="https://example.com/signup",
                text="captcha challenge",
                selector="#submit",
            )
            mem.record_browser_failure(
                tool="browser_act",
                action="click",
                url="https://example.com/signup",
                text="captcha challenge",
                selector="#submit",
            )
            mem.record_browser_success(
                tool="browser_act",
                action="click",
                url="https://example.com/dashboard",
                text="Welcome back",
                selector="button[type=submit]",
            )
            from remy.core.workers.browser_worker import _format_browser_memory_hints

            hints_text = _format_browser_memory_hints(
                {
                    "description": "Sign up on example.com",
                    "target_url": "https://example.com/signup",
                },
            )
        finally:
            mem.settings.DATA_DIR = original

        assert "BROWSER MEMORY" in hints_text
        assert "KNOWN BLOCKER: captcha" in hints_text
        assert "AVOID selectors" in hints_text
        assert "#submit" in hints_text
        assert "PREFER selectors" in hints_text
        assert "button[type=submit]" in hints_text

    def test_prompt_includes_memory_hints_from_session_log(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            mem.record_browser_success(
                tool="browser_act",
                action="click",
                url="https://test.com/dashboard",
                text="Dashboard loaded",
                selector=".btn-login",
            )
            from remy.core.workers.browser_worker import _format_browser_memory_hints

            session_log = [
                {
                    "type": "tool_call",
                    "tool": "browse_page",
                    "evidence": {"page_url": "https://test.com/login"},
                },
            ]
            hints_text = _format_browser_memory_hints(
                {"description": "Login to test.com"},
                session_log=session_log,
            )
        finally:
            mem.settings.DATA_DIR = original

        assert "KNOWN SUCCESS" in hints_text

    def test_prompt_empty_when_no_memory(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            from remy.core.workers.browser_worker import _format_browser_memory_hints

            hints_text = _format_browser_memory_hints(
                {"description": "Visit new-site.com", "target_url": "https://new-site.com"},
            )
        finally:
            mem.settings.DATA_DIR = original

        assert hints_text == ""

    def test_build_prompt_includes_memory_section(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            mem.record_browser_failure(
                tool="browser_act",
                action="click",
                url="https://example.com/signup",
                text="timeout",
                selector="#bad",
            )
            from remy.core.workers.browser_worker import build_browser_worker_prompt

            prompt = build_browser_worker_prompt(
                {
                    "description": "Sign up",
                    "goal_template": "signup_operator",
                    "target_url": "https://example.com/signup",
                },
            )
        finally:
            mem.settings.DATA_DIR = original

        assert "BROWSER MEMORY" in prompt
        assert "BROWSER_WORKER" in prompt


# ============== Orchestrator Domain Blocker History ==============


class TestOrchestratorDomainBlockerHistory:
    def test_escalates_on_repeated_hard_blocker(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            for _ in range(4):
                mem.record_browser_failure(
                    tool="browse_page",
                    url="https://blocked-site.com/signup",
                    text="captcha challenge",
                )
            from remy.core.orchestrator import _check_domain_blocker_history

            result = _check_domain_blocker_history(
                {
                    "goal_template": "signup_operator",
                    "target_url": "https://blocked-site.com/signup",
                },
            )
        finally:
            mem.settings.DATA_DIR = original

        assert result is not None
        assert "captcha" in result["reason"]
        assert "blocked-site.com" in result["reason"]

    def test_no_escalation_below_threshold(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            mem.record_browser_failure(
                tool="browse_page",
                url="https://ok-site.com/signup",
                text="captcha challenge",
            )
            from remy.core.orchestrator import _check_domain_blocker_history

            result = _check_domain_blocker_history(
                {"goal_template": "signup_operator", "target_url": "https://ok-site.com/signup"},
            )
        finally:
            mem.settings.DATA_DIR = original

        assert result is None

    def test_no_escalation_for_non_hard_blocker(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            for _ in range(5):
                mem.record_browser_failure(
                    tool="browser_act",
                    url="https://example.com/signup",
                    text="timeout selecting element",
                )
            from remy.core.orchestrator import _check_domain_blocker_history

            result = _check_domain_blocker_history(
                {"goal_template": "signup_operator", "target_url": "https://example.com/signup"},
            )
        finally:
            mem.settings.DATA_DIR = original

        assert result is None

    def test_detect_external_blocker_uses_history(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            for _ in range(4):
                mem.record_browser_failure(
                    tool="browse_page",
                    url="https://hard-block.com/register",
                    text="email verification required",
                )
            from remy.core.orchestrator import detect_external_blocker

            result = detect_external_blocker(
                {
                    "goal_template": "signup_operator",
                    "target_url": "https://hard-block.com/register",
                },
                [],  # empty session log
            )
        finally:
            mem.settings.DATA_DIR = original

        assert result is not None
        assert "email_verification" in result["reason"]


# ============== Task Metrics: Memory Fields ==============


class TestTaskMetricsMemoryFields:
    def _make_tracker(self, tmp_path):
        from remy.core.task_metrics import TaskMetricsTracker

        return TaskMetricsTracker(path=tmp_path)

    def test_memory_assisted_tracking(self, tmp_path):
        from remy.core.task_metrics import CycleOutcome

        tracker = self._make_tracker(tmp_path)
        tracker.record(CycleOutcome(family="signup_operator", success=True, memory_assisted=True))
        tracker.record(CycleOutcome(family="signup_operator", success=True, memory_assisted=False))
        stats = tracker.get_family("signup_operator")
        assert stats["memory_assisted"] == 1
        assert stats["memory_assist_rate"] == 0.5

    def test_retry_shaped_tracking(self, tmp_path):
        from remy.core.task_metrics import CycleOutcome

        tracker = self._make_tracker(tmp_path)
        tracker.record(CycleOutcome(family="publisher", success=False, retry_shaped=True))
        tracker.record(CycleOutcome(family="publisher", success=True, retry_shaped=True))
        tracker.record(CycleOutcome(family="publisher", success=True))
        stats = tracker.get_family("publisher")
        assert stats["retry_shaped"] == 2
        assert stats["retry_shaped_rate"] == round(2 / 3, 3)

    def test_memory_fields_in_get_all(self, tmp_path):
        from remy.core.task_metrics import CycleOutcome

        tracker = self._make_tracker(tmp_path)
        tracker.record(CycleOutcome(family="signup_operator", success=True, memory_assisted=True))
        tracker.record(
            CycleOutcome(family="publisher", success=True, memory_assisted=True, retry_shaped=True)
        )
        result = tracker.get_all()
        assert result["totals"]["memory_assisted"] == 2
        assert result["totals"]["retry_shaped"] == 1

    def test_memory_fields_persist(self, tmp_path):
        from remy.core.task_metrics import CycleOutcome

        tracker1 = self._make_tracker(tmp_path)
        tracker1.record(
            CycleOutcome(
                family="signup_operator", success=True, memory_assisted=True, retry_shaped=True
            )
        )
        tracker1.flush()

        tracker2 = self._make_tracker(tmp_path)
        stats = tracker2.get_family("signup_operator")
        assert stats["memory_assisted"] == 1
        assert stats["retry_shaped"] == 1

    def test_detect_memory_signals(self):
        from remy.core.task_metrics import detect_memory_signals

        log = [
            {"type": "tool_call", "tool": "browse_page"},
            {"type": "tool_call", "tool": "browser_act", "execution_memory": {"domain": "x.com"}},
            {"type": "tool_call", "tool": "browser_act", "retry_shaped": True},
        ]
        mem_assisted, retry_shaped = detect_memory_signals(log)
        assert mem_assisted is True
        assert retry_shaped is True

    def test_detect_memory_signals_empty(self):
        from remy.core.task_metrics import detect_memory_signals

        mem_assisted, retry_shaped = detect_memory_signals([])
        assert mem_assisted is False
        assert retry_shaped is False

    def test_detect_memory_signals_no_memory(self):
        from remy.core.task_metrics import detect_memory_signals

        log = [{"type": "tool_call", "tool": "browse_page"}]
        mem_assisted, retry_shaped = detect_memory_signals(log)
        assert mem_assisted is False
        assert retry_shaped is False


# ============== Flow Sequence Memory ==============


class TestFlowSequenceMemory:
    def test_record_and_retrieve_signup_flow(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            mem.record_flow_sequence(
                domain="example.com",
                flow="signup",
                steps=[
                    {
                        "tool": "browse_page",
                        "action": "",
                        "selector": "",
                        "url": "https://example.com/signup",
                        "status": "ok",
                    },
                    {
                        "tool": "browser_act",
                        "action": "type",
                        "selector": "#email",
                        "url": "",
                        "status": "ok",
                    },
                    {
                        "tool": "browser_act",
                        "action": "click",
                        "selector": "button[type=submit]",
                        "url": "",
                        "status": "verified",
                    },
                ],
            )
            seq = mem.get_flow_sequence("example.com", "signup")
        finally:
            mem.settings.DATA_DIR = original

        assert seq is not None
        assert seq["domain"] == "example.com"
        assert seq["flow"] == "signup"
        assert len(seq["steps"]) == 3
        assert seq["count"] == 1

    def test_upserts_on_same_domain_flow(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            for _ in range(3):
                mem.record_flow_sequence(
                    domain="example.com",
                    flow="signup",
                    steps=[
                        {
                            "tool": "browse_page",
                            "action": "",
                            "selector": "",
                            "url": "https://example.com/signup",
                            "status": "ok",
                        }
                    ],
                )
            seq = mem.get_flow_sequence("example.com", "signup")
        finally:
            mem.settings.DATA_DIR = original

        assert seq["count"] == 3

    def test_publish_flow_recorded(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            mem.record_flow_sequence(
                domain="blog.com",
                flow="publish",
                steps=[
                    {
                        "tool": "browser_act",
                        "action": "click",
                        "selector": ".publish-btn",
                        "url": "https://blog.com/compose",
                        "status": "verified",
                    }
                ],
            )
            seq = mem.get_flow_sequence("blog.com", "publish")
        finally:
            mem.settings.DATA_DIR = original

        assert seq is not None
        assert seq["flow"] == "publish"

    def test_ignores_non_signup_publish_flows(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            mem.record_flow_sequence(
                domain="example.com",
                flow="navigation",
                steps=[{"tool": "browse_page"}],
            )
            seq = mem.get_flow_sequence("example.com", "navigation")
        finally:
            mem.settings.DATA_DIR = original

        assert seq is None

    def test_ignores_empty_steps(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            mem.record_flow_sequence(domain="example.com", flow="signup", steps=[])
            seq = mem.get_flow_sequence("example.com", "signup")
        finally:
            mem.settings.DATA_DIR = original

        assert seq is None

    def test_get_nonexistent_returns_none(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            seq = mem.get_flow_sequence("nope.com", "signup")
        finally:
            mem.settings.DATA_DIR = original

        assert seq is None

    def test_flow_sequence_in_worker_prompt(self, tmp_path):
        from remy.core import browser_failure_memory as mem

        original = mem.settings.DATA_DIR
        mem.settings.DATA_DIR = tmp_path
        try:
            mem.record_flow_sequence(
                domain="example.com",
                flow="signup",
                steps=[
                    {
                        "tool": "browse_page",
                        "action": "",
                        "selector": "",
                        "url": "https://example.com/signup",
                        "status": "ok",
                    },
                    {
                        "tool": "browser_act",
                        "action": "click",
                        "selector": "#submit",
                        "url": "",
                        "status": "verified",
                    },
                ],
            )
            # Also need a success/failure record so hints aren't empty
            mem.record_browser_success(
                tool="browser_act",
                action="click",
                url="https://example.com/dashboard",
                text="Dashboard",
                selector="#submit",
            )
            from remy.core.workers.browser_worker import _format_browser_memory_hints

            hints = _format_browser_memory_hints(
                {
                    "description": "Sign up on example.com",
                    "target_url": "https://example.com/signup",
                },
            )
        finally:
            mem.settings.DATA_DIR = original

        assert "VERIFIED FLOW" in hints
        assert "signup" in hints
        assert "browse_page" in hints
