"""Browser regression coverage for large Trajectory histories."""

import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture
def trajectory_page(server_url):
    """Own browser fixture so the regression does not depend on pytest-playwright."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.add_init_script(
            "localStorage.setItem('remy_first_run_done_v1', '1');"
        )
        page.goto(server_url, wait_until="domcontentloaded")
        page.wait_for_selector(".sidebar", timeout=10_000)
        page.wait_for_selector("#startup-splash.is-hidden", timeout=10_000)
        yield page
        browser.close()


@pytest.mark.e2e
def test_large_trajectory_keeps_dom_bounded_and_reaches_tail(trajectory_page):
    page = trajectory_page
    record_count = 5_000
    page.evaluate(
        """
        ({ recordCount }) => {
            const records = Array.from({ length: recordCount }, (_, index) => ({
                event_id: `perf-event-${index}`,
                sequence: index + 1,
                kind: ["USER", "REQUEST", "ASSISTANT", "TOOL"][index % 4],
                status: "completed",
                turn_id: `perf-turn-${Math.floor(index / 8)}`,
                duration_ms: index % 17,
                input: index % 4 === 0 ? `Question ${index}` : null,
                output: index % 4 === 2 ? `Answer ${index}` : null,
                details: index % 4 === 3
                    ? { name: "perf_tool", preview: `Result ${index}` }
                    : {},
            }));
            const payload = {
                records,
                summary: {
                    duration_ms: 5000,
                    turns: Math.ceil(recordCount / 8),
                    tool_calls: Math.floor(recordCount / 4),
                },
                pagination: {
                    returned: recordCount,
                    estimated_total: recordCount,
                    has_more: false,
                    window_truncated: false,
                },
                diagnostics: {
                    timing_breakdown: {}, findings: [], error_clusters: [], recovery_paths: [],
                },
                turns: [],
            };
            window.apiClient.getConversationTrajectory = async () => payload;
            document.dispatchEvent(new CustomEvent("conversation-changed", {
                detail: { conversation: { conversation_id: "trajectory-perf" } },
            }));
        }
        """,
        {"recordCount": record_count},
    )

    started_at = page.evaluate("performance.now()")
    page.click("#chat-surface-trajectory")
    page.wait_for_selector("#trajectory-ledger.virtualized", timeout=5_000)
    mounted_in_ms = page.evaluate("performance.now()") - started_at

    ledger_rows = page.locator("#trajectory-ledger .trajectory-record")
    assert mounted_in_ms < 3_000
    assert ledger_rows.count() < 100
    assert page.locator("#trajectory-ledger *").count() < 600
    assert page.locator("#trajectory-ledger .trajectory-ledger-spacer").count() >= 1
    assert page.locator("#trajectory-ledger").get_attribute("aria-rowcount") == str(
        record_count
    )

    page.locator(".trajectory-ledger-wrap").evaluate(
        "element => { element.scrollTop = element.scrollHeight; }"
    )
    page.wait_for_function(
        """() => Array.from(document.querySelectorAll("#trajectory-ledger .trajectory-record"))
            .some(row => row.dataset.eventId === "perf-event-4999")""",
        timeout=5_000,
    )
    assert ledger_rows.count() < 100


@pytest.mark.e2e
def test_virtualized_trajectory_preserves_semantic_selection(trajectory_page):
    page = trajectory_page
    record_count = 1_500
    page.evaluate(
        """
        ({ recordCount }) => {
            const records = Array.from({ length: recordCount }, (_, index) => ({
                event_id: `select-event-${index}`,
                sequence: index + 1,
                kind: "ASSISTANT",
                status: "completed",
                turn_id: `select-turn-${Math.floor(index / 10)}`,
                duration_ms: 1,
                output: `Answer ${index}`,
                details: {},
            }));
            window.apiClient.getConversationTrajectory = async () => ({
                records,
                summary: { duration_ms: recordCount, turns: recordCount / 10, tool_calls: 0 },
                pagination: {
                    returned: recordCount,
                    estimated_total: recordCount,
                    has_more: false,
                    window_truncated: false,
                },
                diagnostics: {
                    timing_breakdown: {}, findings: [], error_clusters: [], recovery_paths: [],
                },
                turns: [],
            });
            document.dispatchEvent(new CustomEvent("conversation-changed", {
                detail: { conversation: { conversation_id: "trajectory-selection" } },
            }));
        }
        """,
        {"recordCount": record_count},
    )

    page.click("#chat-surface-trajectory")
    page.wait_for_selector("#trajectory-ledger.virtualized", timeout=5_000)
    page.locator(".trajectory-ledger-wrap").evaluate(
        "element => { element.scrollTop = element.scrollHeight * 0.5; }"
    )
    page.wait_for_timeout(100)

    middle = page.locator("#trajectory-ledger .trajectory-record").nth(3)
    event_id = middle.get_attribute("data-event-id")
    middle.click()
    selected_class = page.locator(
        f'#trajectory-ledger [data-event-id="{event_id}"]'
    ).get_attribute("class")
    assert "selected" in selected_class
    assert page.locator("#trajectory-inspector").is_visible()


@pytest.mark.e2e
def test_loading_older_trajectory_preserves_the_anchor(trajectory_page):
    page = trajectory_page
    record_count = 1_500
    page.evaluate(
        """
        ({ recordCount }) => {
            const allRecords = Array.from({ length: recordCount }, (_, index) => ({
                event_id: `paged-event-${index}`,
                sequence: index + 1,
                kind: index % 2 ? "ASSISTANT" : "USER",
                status: "completed",
                turn_id: `paged-turn-${Math.floor(index / 6)}`,
                duration_ms: 1,
                input: index % 2 ? null : `Question ${index}`,
                output: index % 2 ? `Answer ${index}` : null,
                details: {},
            }));
            window.apiClient.getConversationTrajectory = async (
                _conversationId, requestedLimit,
            ) => {
                const limit = Math.min(recordCount, Number(requestedLimit || 750));
                const records = allRecords.slice(-limit);
                return {
                    records,
                    summary: { duration_ms: recordCount, turns: recordCount / 6, tool_calls: 0 },
                    pagination: {
                        returned: records.length,
                        estimated_total: recordCount,
                        has_more: limit < recordCount,
                        next_limit: recordCount,
                        window_truncated: limit < recordCount,
                    },
                    diagnostics: {
                        timing_breakdown: {}, findings: [], error_clusters: [], recovery_paths: [],
                    },
                    turns: [],
                };
            };
            document.dispatchEvent(new CustomEvent("conversation-changed", {
                detail: { conversation: { conversation_id: "trajectory-paged" } },
            }));
        }
        """,
        {"recordCount": record_count},
    )

    page.click("#chat-surface-trajectory")
    page.wait_for_selector("#trajectory-load-older:visible", timeout=5_000)
    page.click("#trajectory-load-older")
    page.wait_for_selector(
        '#trajectory-ledger [data-event-id="paged-event-750"]', timeout=5_000
    )
    page.wait_for_timeout(250)

    anchor_delta = page.evaluate(
        """() => {
            const wrap = document.querySelector(".trajectory-ledger-wrap")
                .getBoundingClientRect();
            const anchor = document.querySelector('[data-event-id="paged-event-750"]')
                .getBoundingClientRect();
            return Math.abs(anchor.top - wrap.top);
        }"""
    )
    assert anchor_delta < 80
    assert page.locator("#trajectory-ledger .trajectory-record").count() < 100
