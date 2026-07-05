function esc(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function inlineFormat(text) {
    let html = esc(text);
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    html = html.replace(/\[S(\d+)\]/g, '<a class="research-viewer-citation" href="#source-s$1">[S$1]</a>');
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return html;
}

function renderMarkdown(markdown) {
    const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let inList = false;
    let inCode = false;
    let codeLines = [];
    let paragraph = [];

    const flushParagraph = () => {
        if (!paragraph.length) return;
        out.push(`<p>${inlineFormat(paragraph.join(" "))}</p>`);
        paragraph = [];
    };
    const closeList = () => {
        if (!inList) return;
        out.push("</ul>");
        inList = false;
    };
    const closeCode = () => {
        if (!inCode) return;
        out.push(`<pre><code>${esc(codeLines.join("\n"))}</code></pre>`);
        inCode = false;
        codeLines = [];
    };

    for (const line of lines) {
        if (line.trim().startsWith("```")) {
            flushParagraph();
            closeList();
            if (inCode) closeCode();
            else inCode = true;
            continue;
        }
        if (inCode) {
            codeLines.push(line);
            continue;
        }

        const heading = line.match(/^(#{1,3})\s+(.*)$/);
        if (heading) {
            flushParagraph();
            closeList();
            const level = heading[1].length;
            out.push(`<h${level}>${inlineFormat(heading[2].trim())}</h${level}>`);
            continue;
        }

        const bullet = line.match(/^\s*[-*]\s+(.*)$/);
        if (bullet) {
            flushParagraph();
            if (!inList) {
                out.push("<ul>");
                inList = true;
            }
            out.push(`<li>${inlineFormat(bullet[1].trim())}</li>`);
            continue;
        }

        if (!line.trim()) {
            flushParagraph();
            closeList();
            continue;
        }

        paragraph.push(line.trim());
    }

    flushParagraph();
    closeList();
    closeCode();
    return out.join("\n");
}

function slugify(value) {
    return String(value || "")
        .toLowerCase()
        .replace(/[^a-z0-9\u0400-\u04ff]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 64) || "section";
}

function buildSectionNav(docEl) {
    const sidebar = document.getElementById("research-viewer-sidebar");
    const navEl = document.getElementById("research-viewer-sections");
    if (!sidebar || !navEl || !docEl) return;

    const headings = Array.from(docEl.querySelectorAll("h1, h2, h3"));
    if (!headings.length) {
        sidebar.hidden = true;
        navEl.innerHTML = "";
        return;
    }

    const usedIds = new Set();
    navEl.innerHTML = headings.map((heading) => {
        let id = slugify(heading.textContent || "");
        while (usedIds.has(id)) id += "-x";
        usedIds.add(id);
        heading.id = id;
        return `<a class="research-viewer-section-link research-viewer-section-link--${heading.tagName.toLowerCase()}" href="#${id}">${esc(heading.textContent || "")}</a>`;
    }).join("");
    sidebar.hidden = false;
}

function buildSourceIndex(docEl) {
    const navEl = document.getElementById("research-viewer-sources");
    const wrapperEl = document.getElementById("research-viewer-sources-wrap");
    if (!navEl || !wrapperEl || !docEl) return;

    const headings = Array.from(docEl.querySelectorAll("h1, h2, h3"));
    const sourceHeading = headings.find((heading) => (heading.textContent || "").trim().toLowerCase() === "sources");
    if (!sourceHeading) {
        wrapperEl.hidden = true;
        navEl.innerHTML = "";
        return;
    }

    const sourceItems = [];
    let node = sourceHeading.nextElementSibling;
    while (node && !/^H[1-3]$/.test(node.tagName)) {
        if (node.tagName === "UL") {
            sourceItems.push(...Array.from(node.querySelectorAll("li")));
        }
        node = node.nextElementSibling;
    }

    if (!sourceItems.length) {
        wrapperEl.hidden = true;
        navEl.innerHTML = "";
        return;
    }

    navEl.innerHTML = sourceItems.map((item, index) => {
        const match = (item.textContent || "").match(/\[S(\d+)\]/);
        const label = match ? `S${match[1]}` : `S${index + 1}`;
        const id = `source-${label.toLowerCase()}`;
        item.id = id;
        const itemText = (item.textContent || "").replace(/^\s*\[S\d+\]\s*/, "").trim();
        return `<a class="research-viewer-source-link" href="#${id}"><span class="research-viewer-source-chip">${esc(label)}</span><span>${esc(itemText)}</span></a>`;
    }).join("");
    wrapperEl.hidden = false;
}

async function loadViewer() {
    const shell = document.querySelector(".research-viewer-shell");
    const statusEl = document.getElementById("research-viewer-status");
    const docEl = document.getElementById("research-viewer-document");
    const recordId = shell?.dataset.recordId || "";
    if (!recordId || !statusEl || !docEl) return;

    try {
        const res = await fetch(`/api/autonomy/research-artifacts/${encodeURIComponent(recordId)}/markdown`, {
            credentials: "same-origin",
        });
        if (!res.ok) {
            throw new Error(`Failed with ${res.status}`);
        }
        const markdown = await res.text();
        statusEl.hidden = true;
        docEl.hidden = false;
        docEl.innerHTML = renderMarkdown(markdown);
        buildSectionNav(docEl);
        buildSourceIndex(docEl);
    } catch (err) {
        statusEl.textContent = `Failed to load report: ${err.message || err}`;
        statusEl.classList.add("research-viewer-status--error");
    }
}

loadViewer();
