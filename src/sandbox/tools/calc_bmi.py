import json

TOOL_NAME = "calc_bmi"
TOOL_DESCRIPTION = "Calculate Body Mass Index"
TOOL_PARAMETERS = {
    "height_cm": {"type": "NUMBER", "description": "Height in cm"},
    "weight_kg": {"type": "NUMBER", "description": "Weight in kg"},
}
TOOL_REQUIRED = ["height_cm", "weight_kg"]
DEPENDENCIES = []


def execute(height_cm: float, weight_kg: float) -> str:
    bmi = weight_kg / (height_cm / 100) ** 2
    return json.dumps({"bmi": round(bmi, 1)})


def test_execute():
    result = json.loads(execute(180, 75))
    assert result["bmi"] == 23.1
