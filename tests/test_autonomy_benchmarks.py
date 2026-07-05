import json


def test_run_autonomy_benchmarks_includes_category_summary(tmp_path):
    from remy.core import autonomy_benchmarks as bench

    original_data_dir = bench.settings.DATA_DIR
    bench.settings.DATA_DIR = tmp_path
    try:
        report = bench.run_autonomy_benchmarks()
    finally:
        bench.settings.DATA_DIR = original_data_dir

    assert report["summary"]["total"] >= 6
    assert "execution" in report["category_summary"]
    assert "recovery" in report["category_summary"]
    saved = json.loads((tmp_path / "autonomy_benchmarks.json").read_text(encoding="utf-8"))
    assert saved["summary"] == report["summary"]


def test_load_benchmark_report_returns_none_for_invalid_json(tmp_path):
    from remy.core import autonomy_benchmarks as bench

    original_data_dir = bench.settings.DATA_DIR
    bench.settings.DATA_DIR = tmp_path
    try:
        (tmp_path / "autonomy_benchmarks.json").write_text("{invalid", encoding="utf-8")
        report = bench.load_benchmark_report()
    finally:
        bench.settings.DATA_DIR = original_data_dir

    assert report is None
