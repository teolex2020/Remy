from remy.core.autonomy_live_validation import run_live_validation_pack


def main() -> int:
    report = run_live_validation_pack()
    summary = report["summary"]
    print(
        f"Live validation scenarios: {summary['total']} total | "
        f"ready={summary['ready']} risky={summary['risky']} "
        f"unknown={summary['unknown']} untrained={summary['untrained']}"
    )
    for item in report["results"]:
        print(
            f"[{item['status'].upper()}] {item['name']}: {item['goal_template']} -> {item['target_url'] or 'no-url'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
