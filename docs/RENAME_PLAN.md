# Rename Status

Completed on 2026-07-02.

Current physical layout:

```text
E:\remy\
  app\                # actual git repo and Python package
```

The Python package remains `remy`; imports such as `remy.core.*` do not need to change.

## Current Working Paths

Open the workspace at:

```text
E:\remy
```

Run repo commands from:

```text
E:\remy\app
```

The MCP brain path has been updated to:

```text
E:\remy\app\data\brain
```

## Remaining Sweep Areas

- external shortcuts or scheduled tasks outside the repo;
- IDE workspace files not stored in this repo;
- archived docs that intentionally describe old project history;
- tests that use synthetic old paths as fixture data.

Do not rename `src/remy`; that is the correct import/package identity.
