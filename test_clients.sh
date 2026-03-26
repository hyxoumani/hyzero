#!/bin/bash
# Launches two clients that play a short game (Scholar's Mate)
# White wins in 4 moves: 1. e2e4 e7e5 2. f1c4 b8c6 3. d1h5 g8f6 4. h5f7#

SOCKET="/tmp/hyzero.sock"

# Wait for server socket to appear
echo "Waiting for server..."
for i in $(seq 1 10); do
    [ -S "$SOCKET" ] && break
    sleep 0.5
done
if [ ! -S "$SOCKET" ]; then
    echo "Server socket not found at $SOCKET. Is the server running?"
    exit 1
fi

# Named pipes for capturing client output
WHITE_OUT=$(mktemp)
BLACK_OUT=$(mktemp)

# Start White client in background, feeding moves via a pipe
(
    sleep 0.5  # let both clients connect
    echo "e2e4"
    sleep 1
    echo "f1c4"
    sleep 1
    echo "d1h5"
    sleep 1
    echo "h5f7"
    sleep 1
) | cargo run --bin client 2>&1 > "$WHITE_OUT" &
WHITE_PID=$!

sleep 0.2  # slight delay so White connects first

# Start Black client in background, feeding moves via a pipe
(
    sleep 1.5  # wait for White's first move
    echo "e7e5"
    sleep 1
    echo "b8c6"
    sleep 1
    echo "g8f6"
    sleep 2
) | cargo run --bin client 2>&1 > "$BLACK_OUT" &
BLACK_PID=$!

# Wait for both clients to finish
wait $WHITE_PID 2>/dev/null
wait $BLACK_PID 2>/dev/null

echo ""
echo "=== White Client Output ==="
cat "$WHITE_OUT"
echo ""
echo "=== Black Client Output ==="
cat "$BLACK_OUT"
echo ""

rm -f "$WHITE_OUT" "$BLACK_OUT"
echo "Done. Check server_output.log for server-side move history and game state."
