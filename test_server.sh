#!/bin/bash
# Start the server and log all output to server_output.log
echo "Starting server... (output logged to server_output.log)"
cargo run --bin server 2>&1 | tee server_output.log
