# System Infrastructure

## Local LLM Review Hook

An automatic code review layer using Qwen2.5-Coder:32b via Ollama. Fires after every Edit/Write tool call in Claude Code.

### Components

| Component | Path | Purpose |
|-----------|------|---------|
| Hook script | `~/.claude/hooks/ollama-review.sh` | Sends diffs to Qwen, returns review feedback |
| Hook config | `~/.claude/settings.local.json` | PostToolUse hook registration |
| Ollama | `localhost:11434` | Local inference server |
| Model | `qwen2.5-coder:32b` | 19GB code review model |

### How It Works

```
Claude Edit/Write → PostToolUse fires → hook script reads stdin JSON
  → extracts file path → runs git diff → sends to Qwen via Ollama API
  → if issues found: blocks with review text
  → if LGTM or no diff: exits silently
```

### Hook Input (stdin JSON from Claude Code)

```json
{
  "tool_name": "Edit",
  "tool_input": {
    "file_path": "/path/to/file.rs",
    "old_string": "...",
    "new_string": "..."
  },
  "tool_response": { "success": true }
}
```

### Hook Output (stdout JSON to Claude Code)

When issues found:
```json
{
  "decision": "block",
  "reason": "Qwen Review (file.rs): <review text>"
}
```

When clean: exit 0 with no output (silent pass-through).

### Review Scope

Qwen focuses on **bugs and correctness only**:
- Logic errors
- Off-by-one errors
- Edge cases
- Unsafe patterns
- Type mismatches

### Configuration

- **Reviewed file types**: `.rs`, `.py`, `.ts`, `.js`, `.toml`, `.go`, `.c`, `.cpp`, `.h`, `.java`
- **Max diff size**: 2000 chars (truncated beyond)
- **Timeout**: 55s Ollama call, 60s hook timeout
- **Failure mode**: silent (Ollama down = hook exits 0, no disruption)
- **LGTM behavior**: if Qwen says "LGTM", exits silently (no noise)

### Troubleshooting

| Issue | Fix |
|-------|-----|
| Hook not firing | Check `~/.claude/settings.local.json` has the PostToolUse config |
| Ollama unreachable | Run `ollama serve` or check `curl http://localhost:11434/api/tags` |
| Model not found | Run `ollama pull qwen2.5-coder:32b` |
| Timeout on large diffs | Reduce `MAX_DIFF_CHARS` in the script or increase timeout |
| Want to disable temporarily | Remove or rename `settings.local.json` |
