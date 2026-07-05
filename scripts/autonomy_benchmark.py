from remy.core.autonomy_benchmarks import run_autonomy_benchmarks


def main() -> int:
    report = run_autonomy_benchmarks()
    summary = report["summary"]
    print(
        f"Autonomy benchmarks: {summary['passed']}/{summary['total']} passed "
        f"({summary['pass_rate']}%)"
    )
    for item in report["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['name']}: {item['detail']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
