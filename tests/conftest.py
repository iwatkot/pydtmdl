import os

_PROVIDER_RESULTS: dict[str, str] = {}


def pytest_runtest_logreport(report):
    """Collect per-provider pass/fail results for the GitHub Actions summary."""
    if report.when != "call" or "test_provider[" not in report.nodeid:
        return

    # Node id format: tests/test_providers.py::test_provider[england1m-1024]
    param_str = report.nodeid.split("[", 1)[1].rstrip("]")
    # Strip the trailing size suffix (last segment that is all digits, e.g. "-1024")
    parts = param_str.rsplit("-", 1)
    provider_code = parts[0] if len(parts) == 2 and parts[1].isdigit() else param_str

    if report.passed:
        _PROVIDER_RESULTS[provider_code] = "✅ Passed"
    elif report.failed:
        _PROVIDER_RESULTS[provider_code] = "❌ Failed"
    elif report.skipped:
        _PROVIDER_RESULTS[provider_code] = "⏭️ Skipped"


def pytest_sessionfinish(session, exitstatus):
    """Write a markdown summary table to the GitHub Actions job summary."""
    if not _PROVIDER_RESULTS:
        return

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    passed = sum(1 for v in _PROVIDER_RESULTS.values() if "Passed" in v)
    failed = sum(1 for v in _PROVIDER_RESULTS.values() if "Failed" in v)
    skipped = sum(1 for v in _PROVIDER_RESULTS.values() if "Skipped" in v)

    lines = [
        "## DTM Provider Test Results\n\n",
        f"**{passed} passed** &nbsp;|&nbsp; **{failed} failed** &nbsp;|&nbsp; **{skipped} skipped**\n\n",
        "| Provider | Status |\n",
        "|:---|:---:|\n",
    ]
    for provider_code, status in sorted(_PROVIDER_RESULTS.items()):
        lines.append(f"| `{provider_code}` | {status} |\n")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
