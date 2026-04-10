#!/bin/bash
# Auto-Format — PostToolUse hook for Write|Edit|MultiEdit
# Runs the project's formatter on changed files after each edit.
# Always exits 0 (formatting is advisory, never blocks).

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')

[ -z "$FILE_PATH" ] && exit 0
[ ! -f "$FILE_PATH" ] && exit 0

# Detect formatter based on file extension
case "$FILE_PATH" in
  *.rs)
    command -v rustfmt >/dev/null 2>&1 && rustfmt "$FILE_PATH" 2>/dev/null
    ;;
  *.py)
    command -v ruff >/dev/null 2>&1 && ruff format "$FILE_PATH" 2>/dev/null
    ;;
  *.ts|*.tsx|*.js|*.jsx|*.json|*.css|*.md)
    command -v prettier >/dev/null 2>&1 && prettier --write "$FILE_PATH" 2>/dev/null
    # Fallback to biome
    command -v biome >/dev/null 2>&1 && biome format --write "$FILE_PATH" 2>/dev/null
    ;;
  *.go)
    command -v gofmt >/dev/null 2>&1 && gofmt -w "$FILE_PATH" 2>/dev/null
    ;;
esac

exit 0
