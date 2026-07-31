# Atlas

Autonomous multi-agent deep research platform.

## atlascore

`src/atlascore` is the from-scratch agent framework powering Atlas. This first milestone includes:

- Typed message/event protocol
- OpenAI-compatible chat completion client
- Tool system with `BaseTool` / `FunctionTool`
- Core deterministic tools (calculator, datetime, JSON parser, regex, think, task status)
- `Agent` with `run()` and `run_stream()`

## Quick test

```bash
python -m pytest tests/
```
