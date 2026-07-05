"""
Script to cleanup unwanted 'eval-metric' records from the brain.
Run this once to purge existing garbage.
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remy.config.settings import settings
from aura import Aura as CognitiveMemory

def cleanup():
    # 1. Clean Brain
    print(f"Opening brain at: {settings.AURA_BRAIN_PATH}")
    brain = CognitiveMemory(str(settings.AURA_BRAIN_PATH))
    
    try:
        tag = "eval-metric"
        print(f"Searching BRAIN for records with tag: '{tag}'...")
        
        records = brain.search(query="", tags=[tag], limit=1000)
        
        if records:
            print(f"Found {len(records)} records in Brain.")
            for rec in records:
                try:
                    brain.delete(rec.id)
                    print(f"Deleted from Brain: {rec.id}")
                except Exception as e:
                    print(f"Failed to delete from Brain {rec.id}: {e}")
        else:
            print("Brain is clean.")

    finally:
        brain.close()

    # 2. Clean Knowledge Base (KB)
    if not settings.AURA_MEMORY_ENABLED:
        print("KB disabled in settings.")
        return

    print(f"Opening KB at: {settings.AURA_MEMORY_PATH}")
    try:
        from aura_memory import AuraMemory
        kb = AuraMemory(str(settings.AURA_MEMORY_PATH))
    except Exception as e:
        print(f"Failed to load AuraMemory: {e}")
        return

    # KB doesn't always support tag search, so we iterate and check content/metadata
    # Assuming list_memories returns all or we page through
    print("Scanning KB records for 'Eval: desktop'...")
    
    # We'll fetch a batch. If there are many, we might need a loop.
    # But usually these garbage records are recent.
    memories, total = kb.list_memories(limit=1000)
    
    to_delete = []
    for mem in memories:
        text = mem.get("text", "")
        # Check for specific signature
        if text.startswith("Eval: desktop") or "eval-metric" in mem.get("tags", []):
            to_delete.append(mem["id"])
    
    if not to_delete:
        print("KB is clean.")
        return

    print(f"Found {len(to_delete)} garbage records in KB.")
    confirm = input(f"Delete {len(to_delete)} records from KB? [y/N] ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    count = 0
    for mid in to_delete:
        try:
            kb.delete_memory(mid)
            count += 1
            if count % 10 == 0:
                print(f"Deleted {count} from KB...")
        except Exception as e:
            print(f"Failed to delete {mid}: {e}")

    print(f"Done! Deleted {count} records from KB.")

if __name__ == "__main__":
    cleanup()
