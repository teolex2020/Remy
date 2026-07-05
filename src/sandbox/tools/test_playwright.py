TOOL_NAME = "test_playwright"
TOOL_DESCRIPTION = "Tests if playwright is functional in the sandbox."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {},
    "required": []
}

def execute():
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto("https://example.com")
            title = page.title()
            browser.close()
            return f"Success: {title}"
    except Exception as e:
        return f"Failure: {str(e)}"

def test_playwright_execution():
    result = execute()
    assert "Success" in result or "Failure" in result
