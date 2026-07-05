
import asyncio
from playwright.async_api import async_playwright

TOOL_NAME = "oculus_register"
TOOL_DESCRIPTION = "Automates registration for Oculus Proxies free trial using Playwright."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "email": {"type": "string", "description": "Email address for registration"},
        "password": {"type": "string", "description": "Password for the account"},
        "full_name": {"type": "string", "description": "Full name for the profile"}
    },
    "required": ["email", "password", "full_name"]
}

async def execute(email, password, full_name):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        try:
            # Step 1: Signup page
            await page.goto("https://oculusproxies.com/signup", wait_until="networkidle")
            
            # Step 2: Fill form
            # Inspecting selectors (based on standard signup forms)
            await page.fill('input[name="name"]', full_name)
            await page.fill('input[name="email"]', email)
            await page.fill('input[name="password"]', password)
            await page.fill('input[name="password_confirmation"]', password)
            
            # Step 3: Check for Captcha
            captcha = await page.query_selector('iframe[title*="reCAPTCHA"]')
            if captcha:
                return {"status": "blocked", "reason": "CAPTCHA detected. Need solver."}
            
            # Step 4: Submit
            await page.click('button[type="submit"]')
            await page.wait_for_timeout(5000) # Wait for redirect
            
            # Step 5: Verify success
            if "verify-email" in page.url or "success" in page.content().lower():
                return {"status": "success", "message": "Signup form submitted. Check email."}
            else:
                return {"status": "error", "current_url": page.url, "content_snippet": (await page.content())[:200]}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            await browser.close()

def test_oculus_register():
    # This is a mock test, actual signup requires a real browser
    pass
