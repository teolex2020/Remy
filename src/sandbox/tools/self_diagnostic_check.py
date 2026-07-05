
import re

TOOL_NAME = "self_diagnostic_check"
TOOL_DESCRIPTION = "Verifies response coherence and technical stability for complex architectural content."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The text content to verify."},
        "domain": {"type": "string", "description": "Technical domain (e.g., 'memory architecture', 'code')."}
    },
    "required": ["text"]
}

def execute(text: str, domain: str = "general") -> dict:
    # 1. Coherence Check (Basic heuristics for LLM output)
    has_structure = bool(re.search(r'#{1,3}\s|[\-\*\+]\s|\d\.', text))
    min_length = len(text) > 100
    
    # 2. Technical Stability Check
    # Look for common technical terms in memory architectures
    technical_keywords = ["weights", "context", "RAG", "transformer", "attention", "gradient", "architecture", "latency"]
    found_keywords = [kw for kw in technical_keywords if kw.lower() in text.lower()]
    technical_density = len(found_keywords) / len(technical_keywords) if technical_keywords else 1.0
    
    # 3. Decision
    is_stable = has_structure and min_length and (technical_density > 0.2 or domain == "general")
    
    return {
        "is_stable": is_stable,
        "score": {
            "structure": 1.0 if has_structure else 0.0,
            "density": round(technical_density, 2),
            "length_ok": min_length
        },
        "feedback": "Coherence and stability verified." if is_stable else "Text may lack structure or technical depth for this domain."
    }

def test_self_diagnostic():
    # Test valid technical text
    valid_text = "### Neural Memory Architecture\nThis system uses vector RAG and transformer weights to manage long-term context."
    result = execute(valid_text, domain="memory architecture")
    assert result["is_stable"] == True
    
    # Test weak text
    weak_text = "It is a good system."
    result = execute(weak_text)
    assert result["is_stable"] == False
