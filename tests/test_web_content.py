from remy.core.web_content import extract_visible_text


def test_extract_visible_text_keeps_ssr_content_and_drops_scripts():
    html = """
    <html><head><title>OSMOS — карта професій</title>
    <script>window.fake = 'IPFS Web3 decentralized network'</script></head>
    <body><main><h1>Нова робота починається з твоїх навичок</h1>
    <p>Створи професію та збери практиків.</p></main></body></html>
    """

    text, title = extract_visible_text(html)

    assert title == "OSMOS — карта професій"
    assert "Нова робота" in text
    assert "Створи професію" in text
    assert "IPFS" not in text
