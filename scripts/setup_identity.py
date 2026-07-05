#!/usr/bin/env python3
"""
Setup User Identity for Remy.

Reads all files from the `identity/` folder (txt, md, docx) and stores them
in the agent's long-term memory as IDENTITY-level records. The agent will
know who you are from its very first autonomous cycle.

Usage:
    python scripts/setup_identity.py

Or from Docker:
    docker compose exec remy python scripts/setup_identity.py
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / "src"))

IDENTITY_DIR = project_root / "identity"
SUPPORTED_TEXT = {".txt", ".md"}


def read_docx(path: Path) -> str:
    """Read a .docx file and return plain text (no external deps)."""
    try:
        from zipfile import ZipFile
        from xml.etree import ElementTree

        with ZipFile(path) as zf:
            xml_content = zf.read("word/document.xml")

        tree = ElementTree.fromstring(xml_content)
        ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

        paragraphs = []
        for para in tree.iter(f"{{{ns}}}p"):
            texts = [node.text for node in para.iter(f"{{{ns}}}t") if node.text]
            if texts:
                paragraphs.append("".join(texts))

        return "\n".join(paragraphs)
    except Exception as e:
        print(f"  Warning: Could not read {path.name}: {e}")
        return ""


def read_file(path: Path) -> str:
    """Read a file and return its text content."""
    if path.suffix == ".docx":
        return read_docx(path)
    if path.suffix in SUPPORTED_TEXT:
        return path.read_text(encoding="utf-8")
    return ""


def find_identity_files() -> list[Path]:
    """Find all supported files in identity/ folder."""
    if not IDENTITY_DIR.exists():
        return []
    files = []
    for ext in [*SUPPORTED_TEXT, ".docx"]:
        files.extend(IDENTITY_DIR.glob(f"*{ext}"))
    # Exclude README.txt
    return sorted([f for f in files if f.name != "README.txt"])


def main():
    print("\n=== Remy Identity Setup ===\n")

    files = find_identity_files()
    if not files:
        print(f"No identity files found in: {IDENTITY_DIR}/")
        print("Place a .txt, .md, or .docx file there with your info.")
        print("See identity/README.txt for an example.")
        return

    # Read all files
    all_content = []
    for f in files:
        print(f"  Reading: {f.name}")
        text = read_file(f)
        if text.strip():
            all_content.append(f"--- {f.name} ---\n{text}")

    if not all_content:
        print("All files were empty or unreadable.")
        return

    combined = "\n\n".join(all_content)
    print(f"\nLoaded {len(all_content)} file(s), {len(combined)} chars total.")

    # Import brain
    try:
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")

        from remy.core.agent_tools import brain
        from aura import Level
    except ImportError as e:
        print(f"\nError importing: {e}")
        print("Make sure dependencies are installed (pip install -e .)")
        return

    # Check for existing profile
    existing = brain.search(query="", tags=["user-profile"], limit=5)
    if existing:
        print(f"\nFound {len(existing)} existing profile(s) in brain.")
        choice = input("Overwrite? [y/N] ").strip().lower()
        if choice != "y":
            print("Aborted. Existing profile kept.")
            brain.close()
            return
        for rec in existing:
            brain.delete(rec.id)
            print(f"  Deleted old profile: {rec.id}")

    # Store new profile
    content = f"USER PROFILE (from identity files):\n\n{combined}"

    rec = brain.store(
        content=content,
        level=Level.IDENTITY,
        tags=["user-profile", "identity"],
        metadata={
            "type": "user_profile",
            "source": "setup_identity.py",
            "files": [f.name for f in files],
        },
    )

    print(f"\nStored identity: {rec.id}")
    print(f"Level: IDENTITY (decay 0.99)")
    print(f"Tags: user-profile, identity")
    print(f"\nThe agent will greet you by name on next startup.")

    brain.close()


if __name__ == "__main__":
    main()
