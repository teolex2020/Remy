import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_module():
    path = Path(__file__).resolve().parents[2] / "src" / "sandbox" / "tools" / "capsolver_manager.py"
    spec = importlib.util.spec_from_file_location("capsolver_manager_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_balance():
    module = _load_module()

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"balance": 10.5, "errorId": 0}

        result = module.execute(action="get_balance", api_key="test_key")

    assert result["balance"] == 10.5
    assert mock_post.called


def test_solve_turnstile_create_fail():
    module = _load_module()

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"errorId": 1, "errorCode": "ERROR_KEY_INVALID"}

        result = module.execute(
            action="solve_turnstile",
            api_key="bad_key",
            website_url="http://site.com",
            website_key="site_key",
        )

    assert result["errorId"] == 1
    assert result["errorCode"] == "ERROR_KEY_INVALID"


def test_solve_turnstile_success():
    module = _load_module()

    with patch("requests.post") as mock_post:
        mock_create = MagicMock()
        mock_create.json.return_value = {"errorId": 0, "taskId": "123"}

        mock_result_pending = MagicMock()
        mock_result_pending.json.return_value = {"status": "processing"}

        mock_result_ready = MagicMock()
        mock_result_ready.json.return_value = {
            "status": "ready",
            "solution": {"token": "solved_token"},
        }

        mock_post.side_effect = [mock_create, mock_result_pending, mock_result_ready]

        with patch("time.sleep", return_value=None):
            result = module.execute(
                action="solve_turnstile",
                api_key="key",
                website_url="http://site.com",
                website_key="site_key",
            )

    assert result["status"] == "success"
    assert result["solution"]["token"] == "solved_token"
