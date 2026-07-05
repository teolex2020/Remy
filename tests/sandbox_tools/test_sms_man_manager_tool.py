import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_module():
    path = Path(__file__).resolve().parents[2] / "src" / "sandbox" / "tools" / "sms_man_manager.py"
    spec = importlib.util.spec_from_file_location("sms_man_manager_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sms_man_logic():
    module = _load_module()

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"balance": "10.0"}
        mock_get.return_value = mock_response

        res = module.execute(action="get_balance", api_token="test")
        res_err = module.execute(action="get_number", api_token="test")

    assert res["balance"] == "10.0"
    assert "error" in res_err
