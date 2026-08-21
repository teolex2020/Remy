import json

from remy.core.synaptic_plasticity import EdgeHealth, load_edge_health, save_edge_health


def test_load_edge_health_recovers_trailing_corruption(tmp_path):
    valid = {
        "a|b": {
            "edge_key": "a|b",
            "leak_penalty": 0.2,
            "productive_count": 1,
            "leak_count": 3,
            "conductivity_modifier": 0.8,
            "pruned": False,
            "last_updated": "2026-06-30T00:00:00+00:00",
        }
    }
    path = tmp_path / "edge_health.json"
    path.write_text(json.dumps(valid) + "stale trailing bytes", encoding="utf-8")

    health = load_edge_health(str(tmp_path))

    assert list(health) == ["a|b"]
    assert health["a|b"].leak_count == 3
    assert json.loads(path.read_text(encoding="utf-8")) == valid


def test_save_edge_health_writes_valid_json(tmp_path):
    save_edge_health(
        str(tmp_path),
        {
            "a|b": EdgeHealth(
                edge_key="a|b",
                leak_penalty=0.2,
                productive_count=1,
                leak_count=3,
                conductivity_modifier=0.8,
                pruned=False,
                last_updated="2026-06-30T00:00:00+00:00",
            )
        },
    )

    data = json.loads((tmp_path / "edge_health.json").read_text(encoding="utf-8"))
    assert data["a|b"]["conductivity_modifier"] == 0.8
