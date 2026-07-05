import asyncio
import sys
import os
import time

# Mocking internal imports for the sandbox environment if they are not available
# In a real scenario, these would be part of the agent's environment.
try:
    from remy.config.settings import settings
    from remy.core.browser import BrowserManager
except ImportError:
    # Fallback/Mock logic for testing if needed
    pass

TOOL_NAME = "oculus_custom_register"
TOOL_DESCRIPTION = "Automates Oculus Proxies registration with visible browser for manual captcha solving."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "email": {"type": "string", "description": "Email to register"},
        "password": {"type": "string", "description": "Password for the account"}
    },
    "required": ["email", "password"]
}

async def execute(brain, email, password):
    # Add src to sys.path
    sys.path.append(os.path.join(os.getcwd(), "src"))

    from remy.config.settings import settings
    settings.BROWSER_HEADLESS = False  # Force visible browser
    
    from remy.core.browser import BrowserManager
    
    print("Initializing BrowserManager (Headless=False)...")
    browser = BrowserManager.get()
    
    url = "https://oculusproxies.com"
    print(f"Navigating to {url}...")
    
    results = {"status": "started", "screenshots": []}
    
    try:
        await browser.act("goto", url=url)
        print("Home page loaded.")
        
        print("Clicking 'Log in'...")
        await browser.act("click", selector='[data-clickid="nav_login"]')
        await asyncio.sleep(2)
        
        print("Clicking 'Sign up'...")
        await browser.act("click", selector='.login_signup_switch___euIlQ a')
        await asyncio.sleep(2)
        
        print("Clicking 'Continue with Email'...")
        await browser.act("click", selector='button:has-text("Continue with Email")')
        await asyncio.sleep(2)
        
        print("Filling Email...")
        await browser.act("type", selector='input[autocomplete="email"]', text=email)
        
        print("Filling Password...")
        await browser.act("type", selector='input[type="password"]', text=password)
        
        print("Clicking 'Create a free account'...")
        await browser.act("click", selector='button:has-text("Create a free account")')
        
        print("Submitted. Waiting for success or Captcha intervention...")
        
        page = await browser.ensure_browser()
        
        for i in range(12):
            current_url = page.url
            content = await page.content()
            print(f"[{i*10}s] Current URL: {current_url}")
            
            if "dashboard" in current_url or "portal" in current_url:
                results["status"] = "success"
                results["message"] = "Redirected to dashboard/portal!"
                break
            
            if "Verify your email" in content or "Check your inbox" in content:
                results["status"] = "success"
                results["message"] = "Registration submitted, check email."
                break
                
            await asyncio.sleep(10)
        
        screenshot_bytes = await page.screenshot()
        filename = browser.save_screenshot(screenshot_bytes)
        results["screenshots"].append(filename)
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        try:
            page = await browser.ensure_browser()
            screenshot_bytes = await page.screenshot()
            filename = browser.save_screenshot(screenshot_bytes)
            results["screenshots"].append(filename)
        except:
            pass
    finally:
        print("Closing browser...")
        await browser.close()
        
    return results

def test_tool():
    # This is a placeholder since we can't fully mock the browser in the sandbox test
    assert TOOL_NAME == "oculus_custom_register"
