
import re

TOOL_NAME = "self_diagnostic"
TOOL_DESCRIPTION = "Verifies text coherence and technical stability (e.g., no placeholders, no infinite loops in logic, balanced brackets)."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The text to diagnose."},
        "check_type": {"type": "string", "enum": ["coherence", "code", "architecture"], "default": "coherence"}
    },
    "required": ["text"]
}

def execute(text: str, check_type: str = "coherence") -> str:
    issues = []
    
    # Check for common placeholders
    placeholders = re.findall(r"\[.*?\]|<.*?>|\.\.\.", text)
    if placeholders:
        issues.append(f"Detected potential placeholders: {list(set(placeholders))[:5]}")
        
    # Check for balanced brackets (basic)
    for opening, closing in [('(', ')'), ('{', '}'), ('[', ']')]:
        if text.count(opening) != text.count(closing):
            issues.append(f"Unbalanced brackets detected: {opening} vs {closing}")
            
    # Check for repetitive patterns (potential LLM loop)
    lines = text.split('\n')
    if len(lines) > 10:
        for i in range(len(lines) - 3):
            if lines[i] == lines[i+1] == lines[i+2]:
                issues.append("Repetitive line loop detected.")
                break
                
    if not issues:
        return "PASS: No immediate stability issues detected."
    else:
        return f"FAIL: {'; '.join(issues)}"

def test_self_diagnostic():
    # Test pass
    assert "PASS" in execute("This is a solid architectural plan (v1). { 'key': 'value' }")
    # Test placeholders
    assert "FAIL" in execute("Follow the [INSERT STEP] here.")
    # Test brackets
    assert "FAIL" in execute("Unbalanced ( brackets")
    # Test loops
    assert "FAIL" in execute("Loop\nLoop\nLoop\nLoop")
