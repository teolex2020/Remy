"""Browser contract for the approval-gated Self-improvement Lab surface."""

import pytest
from playwright.sync_api import sync_playwright


@pytest.mark.e2e
def test_self_improvement_lab_renders_release_gates_without_eager_loading(server_url):
    proposal = {
        "proposal_id": "self-mod-ui-test",
        "target": "agent.guidance",
        "candidate_text": (
            "Separate verified evidence from inference before presenting the final answer."
        ),
        "candidate_hash": "a" * 64,
        "baseline_hash": "b" * 64,
        "rationale": "Make evidence boundaries visible.",
        "source": "operator",
        "status": "draft",
        "evaluation": {},
        "canary_evaluation": {},
        "approved_by": "",
        "approved_at": "",
        "updated_at": "2026-08-21T12:00:00+00:00",
        "canary_percent": 0,
        "trajectory_session_id": "self-mod:self-mod-ui-test",
    }
    cases = [
        {
            "case_id": f"case-{index}",
            "name": f"Regression case {index}",
            "latest_run": None,
        }
        for index in range(1, 4)
    ]
    telemetry = {
        "proposal_id": proposal["proposal_id"],
        "status": "collecting",
        "ready": False,
        "gate_passed": None,
        "privacy": "aggregate-only; prompts and responses excluded",
        "candidate": {
            "requests": 3, "completed_requests": 3, "failed_requests": 0,
            "verified_requests": 3, "unsupported_requests": 0,
            "failure_rate": 0, "unsupported_rate": 0,
            "avg_request_ms": 800, "verification_coverage": 1,
        },
        "baseline": {
            "requests": 8, "completed_requests": 8, "failed_requests": 0,
            "verified_requests": 8, "unsupported_requests": 0,
            "failure_rate": 0, "unsupported_rate": 0,
            "avg_request_ms": 900, "verification_coverage": 1,
        },
        "readiness_checks": {
            "enough_candidate_requests": False,
            "enough_baseline_requests": True,
            "candidate_verification_coverage": True,
            "baseline_verification_coverage": True,
        },
        "promotion_readiness_checks": {
            "candidate_sample_target": False,
            "baseline_sample_target": False,
            "minimum_observation_window": False,
            "candidate_verification_coverage": True,
            "baseline_verification_coverage": True,
        },
        "requirements": {
            "minimum_requests_per_cohort": 5,
            "target_requests_per_cohort": 20,
            "minimum_observation_seconds": 300,
            "minimum_verification_coverage": 0.8,
            "confidence_level": 0.95,
        },
        "observation": {"seconds": 90, "minimum_seconds": 300},
        "statistics": {
            "confidence_score": 0.15,
            "confidence_level": 0.95,
            "checks": {
                "failure_rate_non_inferior": True,
                "unsupported_rate_non_inferior": True,
                "latency_non_inferior": True,
            },
            "failure_rate": {"candidate": {"upper": 0.56}, "baseline": {"upper": 0.32}},
            "unsupported_rate": {"candidate": {"upper": 0.56}, "baseline": {"upper": 0.32}},
            "latency_ms": {"candidate": {"upper": 900}, "baseline": {"upper": 980}},
        },
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.add_init_script(
            "localStorage.setItem('remy_first_run_done_v1', '1');"
        )
        self_mod_requests = []
        policy_requests = []
        page.route(
            "**/api/experiments/models",
            lambda route: route.fulfill(json={"models": [{"name": "test-model", "provider": "test"}]}),
        )
        page.route(
            "**/api/experiments/roles",
            lambda route: route.fulfill(json={"roles": []}),
        )
        page.route(
            "**/api/experiments",
            lambda route: route.fulfill(json={"experiments": []}),
        )

        def fulfill_self_modifications(route):
            self_mod_requests.append(route.request.url)
            route.fulfill(json={
                "proposals": [proposal],
                "policy": {
                    "minimum_requests_per_cohort": 5,
                    "target_requests_per_cohort": 20,
                    "minimum_observation_seconds": 300,
                    "minimum_verification_coverage": 0.8,
                    "confidence_level": 0.95,
                    "failure_rate_margin": 0.02,
                    "unsupported_rate_margin": 0.02,
                    "latency_multiplier": 1.5,
                    "inconclusive_alert_seconds": 1800,
                    "alerts_enabled": True,
                    "source": "default",
                },
                "constraints": {
                    "target": "agent.guidance",
                    "additive_only": True,
                    "immutable": ["code", "base_prompt", "tools", "policy", "approval", "sandbox"],
                    "min_eval_cases": 3,
                    "canary_percent_range": [5, 25],
                },
            })

        page.route("**/api/experiments/self-modifications", fulfill_self_modifications)
        def fulfill_policy(route):
            payload = route.request.post_data_json
            policy_requests.append(payload)
            route.fulfill(json={"policy": {**payload, "source": "project"}})

        page.route("**/api/experiments/self-modifications/policy", fulfill_policy)
        page.route(
            "**/api/trajectory/analytics/eval-cases?limit=100",
            lambda route: route.fulfill(json={"cases": cases}),
        )
        page.route(
            "**/api/experiments/self-modifications/self-mod-ui-test/canary-telemetry",
            lambda route: route.fulfill(json={"telemetry": telemetry}),
        )
        page.goto(server_url, wait_until="domcontentloaded")
        page.wait_for_selector(".sidebar", timeout=10_000)
        page.wait_for_selector("#startup-splash.is-hidden", timeout=10_000)
        page.click('[data-view="experiments"]')
        page.wait_for_selector('[data-experiment-section="self-modification"]')

        assert self_mod_requests == []
        page.click('[data-experiment-section="self-modification"]')
        page.wait_for_selector(".self-mod-proposal-detail")

        assert len(self_mod_requests) == 1
        assert page.locator(".self-mod-progress li").count() == 5
        assert page.locator("[data-self-mod-case]:checked").count() == 3
        assert page.get_by_text("Safe self-improvement boundary").is_visible()
        assert page.get_by_text("Canary policy").is_visible()
        page.get_by_text("Canary policy").click()
        page.locator('[name="target_requests_per_cohort"]').fill("30")
        page.get_by_role("button", name="Save project policy").click()
        page.wait_for_timeout(100)
        assert policy_requests[0]["target_requests_per_cohort"] == 30
        assert policy_requests[0]["minimum_verification_coverage"] == 0.8
        assert page.get_by_role("button", name="Run sandbox gate").is_enabled()
        assert page.get_by_role("button", name="Trajectory").is_visible()
        assert page.locator(".self-mod-layout").evaluate(
            "element => element.scrollWidth <= element.clientWidth"
        )

        proposal["status"] = "canary"
        proposal["canary_percent"] = 10
        page.click('[data-experiment-section="experiments"]')
        page.wait_for_selector('[data-experiment-section="self-modification"]')
        page.click('[data-experiment-section="self-modification"]')
        page.wait_for_selector(".self-mod-telemetry")
        assert page.get_by_text("Live Trajectory telemetry").is_visible()
        assert page.get_by_text("3 requests").is_visible()
        assert page.get_by_text("8 requests").is_visible()
        assert page.get_by_text("95% statistical confidence").is_visible()
        assert page.get_by_text("15.0%").is_visible()
        assert page.get_by_role("button", name="Refresh & enforce gate").is_visible()
        assert not page.locator(".self-mod-canary-eval").get_attribute("open")

        page.set_viewport_size({"width": 760, "height": 900})
        page.wait_for_timeout(50)
        assert page.locator(".self-mod-layout").evaluate(
            "element => getComputedStyle(element).gridTemplateColumns.split(' ').length === 1"
        )
        assert page.locator(".self-mod-layout").evaluate(
            "element => element.scrollWidth <= element.clientWidth"
        )
        browser.close()
