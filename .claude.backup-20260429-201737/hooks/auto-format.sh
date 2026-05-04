#!/bin/bash
# Auto-Format — PostToolUse hook for Write|Edit|MultiEdit
# Runs the project's formatter on changed files after each edit.
# Always exits 0 (formatting is advisory, never blocks).

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0
[ ! -f "$FILE_PATH" ] && exit 0

case "$FILE_PATH" in
  *.rs)
    command -v rustfmt >/dev/null 2>&1 && rustfmt "$FILE_PATH" 2>/dev/null
    ;;
  *.py)
    command -v ruff >/dev/null 2>&1 && ruff format "$FILE_PATH" 2>/dev/null
    ;;
  *.ts|*.tsx|*.js|*.jsx|*.json|*.css|*.md)
    if command -v prettier >/dev/null 2>&1; then
      prettier --write "$FILE_PATH" 2>/dev/null
    elif command -v biome >/dev/null 2>&1; then
      biome format --write "$FILE_PATH" 2>/dev/null
    fi
    ;;
  *.go)
    command -v gofmt >/dev/null 2>&1 && gofmt -w "$FILE_PATH" 2>/dev/null
    ;;
esac

exit 0
