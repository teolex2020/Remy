
import json
import re

TOOL_NAME = "memory_diag_tool"
TOOL_DESCRIPTION = "Validates neural network memory architecture coherence and technical stability (Aura Cognitive standard)."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "architecture_json": {
            "type": "string",
            "description": "JSON representation of the memory architecture (nodes, levels, links)."
        },
        "diagram_mermaid": {
            "type": "string",
            "description": "Optional Mermaid code for the architecture diagram to check syntax."
        }
    },
    "required": ["architecture_json"]
}

def check_causal_cycles(nodes, links):
    # Simple DFS to detect cycles in 'depends_on' relations
    adj = {node['id']: [] for node in nodes}
    for link in links:
        if link['type'] == 'depends_on':
            adj[link['source']].append(link['target'])
    
    visited = set()
    path = set()
    
    def has_cycle(v):
        visited.add(v)
        path.add(v)
        for neighbor in adj.get(v, []):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in path:
                return True
        path.remove(v)
        return False
    
    for node in nodes:
        if node['id'] not in visited:
            if has_cycle(node['id']):
                return True
    return False

def execute(architecture_json: str, diagram_mermaid: str = None) -> str:
    results = {"valid": True, "errors": [], "warnings": []}
    
    # 1. JSON Parsing
    try:
        data = json.loads(architecture_json)
    except Exception as e:
        return json.dumps({"valid": False, "errors": [f"Invalid JSON: {str(e)}"]})
    
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    
    # 2. Hierarchy Check (L1-L4)
    valid_levels = {"L1_WORKING", "L2_DECISIONS", "L3_DOMAIN", "L4_IDENTITY"}
    for node in nodes:
        if node.get("level") not in valid_levels:
            results["errors"].append(f"Node {node.get('id')} has invalid level: {node.get('level')}")
            results["valid"] = False
            
    # 3. Causal Cycle Detection
    if check_causal_cycles(nodes, links):
        results["errors"].append("Causal dependency cycle detected (depends_on loop).")
        results["valid"] = False
        
    # 4. Diagram Syntax Check (Very basic)
    if diagram_mermaid:
        if not diagram_mermaid.strip().startswith(("graph", "flowchart", "sequenceDiagram", "classDiagram", "stateDiagram")):
            results["warnings"].append("Mermaid diagram does not start with a valid type (e.g., 'graph TD').")
        if "-->" not in diagram_mermaid and "->" not in diagram_mermaid:
             results["warnings"].append("Mermaid diagram seems to lack connections.")

    # 5. Metric Validation
    for node in nodes:
        surprise = node.get("metrics", {}).get("surprise")
        if surprise is not None and not (0.0 <= surprise <= 1.0):
            results["errors"].append(f"Node {node.get('id')} has invalid surprise metric: {surprise}")
            results["valid"] = False

    return json.dumps(results, indent=2)

def test_memory_diag():
    # Test valid
    valid_arch = {
        "nodes": [
            {"id": "node1", "level": "L1_WORKING", "metrics": {"surprise": 0.5}},
            {"id": "node2", "level": "L2_DECISIONS"}
        ],
        "links": [
            {"source": "node1", "target": "node2", "type": "depends_on"}
        ]
    }
    res = json.loads(execute(json.dumps(valid_arch)))
    assert res["valid"] == True
    
    # Test cycle
    cycle_arch = {
        "nodes": [{"id": "A", "level": "L1_WORKING"}, {"id": "B", "level": "L1_WORKING"}],
        "links": [
            {"source": "A", "target": "B", "type": "depends_on"},
            {"source": "B", "target": "A", "type": "depends_on"}
        ]
    }
    res = json.loads(execute(json.dumps(cycle_arch)))
    assert res["valid"] == False
    assert "Causal dependency cycle detected" in res["errors"][0]
