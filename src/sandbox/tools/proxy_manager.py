
import json
import random

TOOL_NAME = "proxy_manager"
TOOL_DESCRIPTION = "Manages a pool of proxies for autonomous web navigation. Supports adding, retrieving, and rotating proxies to avoid IP-based blocking and anti-bot systems."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": "The action to perform: 'add', 'get_random', 'list', 'delete', 'rotate'.",
            "enum": ["add", "get_random", "list", "delete", "rotate"]
        },
        "proxy_url": {
            "type": "string",
            "description": "Proxy URL in format http://user:pass@host:port (required for 'add')."
        },
        "proxy_id": {
            "type": "string",
            "description": "The ID of the proxy record (required for 'delete')."
        },
        "tags": {
            "type": "string",
            "description": "Optional tags for categorization (e.g., 'residential', 'country:UA')."
        }
    },
    "required": ["action"]
}

def execute(brain, action: str, proxy_url: str = None, proxy_id: str = None, tags: str = None) -> str:
    if action == "add":
        if not proxy_url:
            return "Error: proxy_url is required for 'add' action."
        
        record_tags = "proxy,technical"
        if tags:
            record_tags += f",{tags}"
        
        existing = brain.search(proxy_url, tags="proxy")
        if existing:
            return f"Proxy already exists in memory. ID: {existing[0]['id']}"
            
        result_id = brain.store(content=f"Proxy: {proxy_url}", level="L3_DOMAIN", tags=record_tags)
        return f"Proxy added successfully. ID: {result_id}"

    elif action == "list":
        proxies = brain.search("", tags="proxy")
        if not proxies:
            return "No proxies found in memory."
        return json.dumps(proxies, indent=2)

    elif action == "get_random":
        proxies = brain.search("", tags="proxy")
        if not proxies:
            return "Error: No proxies available. Add a proxy first."
        selected = random.choice(proxies)
        return json.dumps(selected, indent=2)

    elif action == "delete":
        if not proxy_id:
            return "Error: proxy_id is required for 'delete' action."
        brain.delete(proxy_id)
        return f"Proxy {proxy_id} deleted."

    elif action == "rotate":
        proxies = brain.search("", tags="proxy")
        if not proxies:
            return "Error: No proxies available for rotation."
        selected = random.choice(proxies)
        return f"Rotated to proxy: {selected['content']}"

    return "Invalid action."

def test_proxy_manager():
    class MockBrain:
        def __init__(self):
            self.data = {}
            self.counter = 0
        def store(self, content, level, tags):
            self.counter += 1
            idx = str(self.counter)
            self.data[idx] = {"id": idx, "content": content, "tags": tags}
            return idx
        def search(self, query, tags):
            return [v for v in self.data.values() if tags in v['tags']]
        def delete(self, idx):
            if idx in self.data: del self.data[idx]

    brain = MockBrain()
    res = execute(brain, action="add", proxy_url="http://user:pass@1.1.1.1:8080", tags="test")
    assert "added successfully" in res
    res = execute(brain, action="list")
    assert "1.1.1.1" in res
    res = execute(brain, action="get_random")
    assert "1.1.1.1" in res
    id_to_del = json.loads(execute(brain, action="list"))[0]['id']
    res = execute(brain, action="delete", proxy_id=id_to_del)
    assert "deleted" in res
    print("All tests passed!")

if __name__ == "__main__":
    test_proxy_manager()
