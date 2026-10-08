# dualmem

A memory layer for LLM agents with two cooperating stores:

- **Episodes** — atomic, timestamped events, each linked to its source messages.
- **Facts** — durable statements with validity windows, so changed information is superseded, never silently overwritten.

Every memory is traceable back to the conversation that produced it (provenance), and a consolidation pass promotes recurring evidence from episodes into durable facts.

> **Status: early development (v0).** API will change.

## Install

```bash
pip install dualmemory
```

## License

MIT
