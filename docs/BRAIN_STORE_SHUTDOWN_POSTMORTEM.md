# Brain Store Shutdown Postmortem

Date: 2026-03-23
Status: mitigated

## Summary

Remy could lose large visible portions of memory after restart even though the user had continued working for days after a clean reset.

The issue was not a normal forgetting path.
It was a persistence-lifecycle failure during shutdown that could corrupt the active Aura brain store, trigger startup quarantine, and make Remy reopen on a fresh empty store.

## What Broke

Observed symptoms:

- `Memory` suddenly showed `0 records`
- `Graph` showed `0 nodes`
- `Profile` and `People` appeared empty
- `History` still contained prior conversations and tool calls

This created the impression that the agent had "forgotten everything", but the actual failure mode was:

1. active brain store was interrupted during shutdown
2. next startup probe detected the store as unreadable
3. Remy quarantined the store
4. Remy reopened on a fresh empty brain path
5. UI surfaces then reflected the empty active brain

## Root Cause

Primary root cause:

- the old Windows shutdown path in [`src/remy/core/combined_runner.py`](/E:/remy/app/src/remy/core/combined_runner.py) used `os._exit(1)` after a second `Ctrl+C`

Why that was dangerous:

- it bypassed normal `finally` cleanup
- it bypassed explicit `brain.close()`
- it could terminate the process while Aura persistence was still flushing

In logs this appeared as:

- graceful shutdown start
- immediate second interrupt / force quit
- next startup reporting unreadable store data

This was therefore an integration bug between:

- Remy shutdown/runtime handling
- Aura store persistence lifecycle

It was not primarily a UI bug and not ordinary memory decay.

## Why History Survived

[`data/history`](/E:/remy/app/data/history) is separate from the active Aura brain store.

So after corruption:

- history logs still existed
- active brain-backed surfaces were empty

That separation made recovery possible.

## Fixes Already Applied

### 1. Safer shutdown

In [`src/remy/core/combined_runner.py`](/E:/remy/app/src/remy/core/combined_runner.py):

- removed the dangerous `os._exit(1)` path on second `Ctrl+C`
- second interrupt now requests a faster shutdown instead of bypassing cleanup
- shutdown messaging now explicitly tells the operator not to interrupt again

### 2. Explicit brain close

In [`src/remy/core/agent_tools.py`](/E:/remy/app/src/remy/core/agent_tools.py):

- explicit `close_brain()` is registered on exit
- successful close is logged as `Brain closed cleanly`

### 3. Startup recovery after quarantine

In:

- [`src/remy/core/agent_tools.py`](/E:/remy/app/src/remy/core/agent_tools.py)
- [`src/remy/core/history_replay.py`](/E:/remy/app/src/remy/core/history_replay.py)
- [`scripts/replay_history_to_brain.py`](/E:/remy/app/scripts/replay_history_to_brain.py)

Remy now:

- detects when startup quarantine happened
- checks whether the fresh active brain is empty
- automatically replays safe memory-writing tool calls from history

This does not restore everything perfectly, but it prevents the worst-case production behavior of silently coming up empty.

## Recovery Strategy

Current recovery order:

1. try to open the existing Aura store safely
2. if unreadable, quarantine it rather than deleting it
3. start on a fresh store
4. auto-replay safe history calls into the fresh store

Replay scope is intentionally bounded to safe memory-writing tools:

- `store`
- `store_person`
- `store_research`
- `store_story`
- `store_user_profile`
- `schedule_task`

Failed tool calls and validation-error calls are skipped.

## Remaining Production Guardrails Worth Adding

### High priority

- emit a visible operator alert when startup recovery runs
- persist a recovery incident marker in logs and system status
- add a startup integrity summary in `System`
- add a shutdown state indicator that clearly shows when it is safe to restart

### Medium priority

- add periodic snapshots / rollback points around active brain state
- add startup comparison checks:
  - previous record count
  - recovered record count
  - quarantine reason
- add an automated regression test for interrupt-during-shutdown behavior

### Longer term

- improve Aura-side resilience to abrupt process death
- add a more formal export/import path for critical identity and operator data
- distinguish "store format mismatch" from "store corruption" in startup diagnostics

## Operational Guidance

Recommended stop procedure during development:

1. press `Ctrl+C` once or use `Stop Server`
2. wait for graceful shutdown to finish
3. do not press `Ctrl+C` again
4. only restart after the process fully exits

If a quarantine still happens:

1. do not delete quarantine folders immediately
2. check logs for the quarantine reason
3. allow startup recovery to replay history
4. verify `Memory`, `Profile`, and `Graph`

## Outcome

This incident showed that the main failure was not "the memory forgot".

The real problem was:

- unsafe shutdown
- corrupted active store
- empty fresh restart without automatic recovery

That path is now mitigated.
