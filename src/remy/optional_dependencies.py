"""Actionable checks for Remy's optional feature packages."""

from importlib.util import find_spec


def require_extra(module_name: str, extra: str, feature: str) -> None:
    """Exit cleanly when a command needs an extra that is not installed."""
    if find_spec(module_name) is not None:
        return
    raise SystemExit(
        f"Remy {feature} requires optional dependencies. "
        f'Install them with: pip install "remy[{extra}]"'
    )
