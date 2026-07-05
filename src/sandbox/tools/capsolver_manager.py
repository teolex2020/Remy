import time
import requests

TOOL_NAME = "capsolver_manager"
TOOL_DESCRIPTION = "Bypass CAPTCHAs and anti-bot challenges using CapSolver API (2026). Supports Cloudflare Turnstile, reCAPTCHA v3/Enterprise, and others."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": "The action to perform: 'get_balance', 'solve_turnstile', 'solve_recaptcha_v3'",
            "enum": ["get_balance", "solve_turnstile", "solve_recaptcha_v3"]
        },
        "api_key": {
            "type": "string",
            "description": "CapSolver API Key"
        },
        "website_url": {
            "type": "string",
            "description": "URL of the website with CAPTCHA"
        },
        "website_key": {
            "type": "string",
            "description": "Site key for CAPTCHA"
        },
        "page_action": {
            "type": "string",
            "description": "Action name for reCAPTCHA v3 (e.g., 'login', 'homepage')"
        },
        "metadata": {
            "type": "object",
            "description": "Optional metadata for Turnstile (cdata, action)"
        }
    },
    "required": ["action", "api_key"]
}

def execute(action, api_key, website_url=None, website_key=None, page_action=None, metadata=None):
    base_url = "https://api.capsolver.com"
    
    if action == "get_balance":
        try:
            response = requests.post(f"{base_url}/getBalance", json={"clientKey": api_key}, timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    task_payload = {
        "clientKey": api_key,
        "task": {}
    }
    
    if action == "solve_turnstile":
        if not website_url or not website_key:
            return {"error": "website_url and website_key are required for Turnstile"}
        task_payload["task"] = {
            "type": "AntiTurnstileTaskProxyLess",
            "websiteURL": website_url,
            "websiteKey": website_key,
            "metadata": metadata or {}
        }
    
    elif action == "solve_recaptcha_v3":
        if not website_url or not website_key or not page_action:
            return {"error": "website_url, website_key, and page_action are required for reCAPTCHA v3"}
        task_payload["task"] = {
            "type": "ReCaptchaV3TaskProxyLess",
            "websiteURL": website_url,
            "websiteKey": website_key,
            "pageAction": page_action,
            "minScore": 0.7
        }
    
    try:
        # Create Task
        res = requests.post(f"{base_url}/createTask", json=task_payload, timeout=10)
        res_data = res.json()
        
        if res_data.get("errorId") != 0:
            return res_data
            
        task_id = res_data.get("taskId")
        
        # Poll for result
        for _ in range(15): # 30 seconds max (15 * 2)
            time.sleep(2)
            result = requests.post(f"{base_url}/getTaskResult", json={
                "clientKey": api_key,
                "taskId": task_id
            }, timeout=10).json()
            
            if result.get("status") == "ready":
                return {"status": "success", "solution": result.get("solution")}
            
            if result.get("status") == "failed":
                return {"status": "failed", "error": result.get("errorDescription")}
                
        return {"status": "timeout", "message": "Task did not complete in time"}
    except Exception as e:
        return {"error": str(e)}
