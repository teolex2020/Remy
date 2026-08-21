# Remy Product Architecture

Remy should be packaged as a local AI workflow automation app, not a chat-first assistant.

## Product Surface

The first screen should be:

1. Home
2. Template Gallery
3. Run Template
4. Dry Run
5. Run History
6. Settings / Local Secrets

Chat is a tool for asking questions and inspecting workflow output. It is not the primary product surface.

## Runtime Layers

```text
Desktop / Web Shell
  -> Home + Templates + Run History
  -> Workflow runner / pipeline runner
  -> Human approval + dry-run gate
  -> Tool execution + integrations
  -> Reliability layer
       - consequence memory
       - epistemic governance
       - admission classes
       - factuality verification
  -> Aura memory adapter
```

## Packaging Rules

- End-user distribution should be `Remy`; the repo lives at `E:\remy\app`.
- The repo package name is already `remy`; keep Python imports as `remy.*`.
- Keep lab/debug views available, but move them behind Advanced/Lab language.
- Product copy should say "local AI workflows", "templates", "run history", and "human approval".
- Technical copy may say "trustworthy autonomous runtime" only in architecture, grants, or developer docs.

## First Paid Surface

Free core:

- local app
- chat/ask
- memory
- basic workflow templates
- dry run and run history

Paid or donation-worthy later:

- Windows installer / portable build
- extra workflow packs
- pro workflow blocks
- support / priority builds
