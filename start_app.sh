#!/bin/bash
# 🚀 Start Similarity Search App + Tunnel
# This script starts the Streamlit app and a public tunnel.
# Runs via launchd on boot.

APP_DIR="$HOME/Desktop/similarity-search-app"
PORT=8511
LOG_DIR="$HOME/Library/Logs/similarity-app"

mkdir -p "$LOG_DIR"

# Set PATH
export PATH="$HOME/Library/Python/3.9/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

echo "=== Starting Similarity Search App ===" >> "$LOG_DIR/startup.log"
date >> "$LOG_DIR/startup.log"

# Kill any previous instances
pkill -f "streamlit run.*similarity_app" 2>/dev/null
pkill -f "ssh.*serveo.net" 2>/dev/null

# Wait for cleanup
sleep 2

# Start Streamlit app
cd "$APP_DIR"
nohup streamlit run catalog_app.py --server.port $PORT --server.headless true \
    >> "$LOG_DIR/streamlit.log" 2>&1 &
STREAMLIT_PID=$!
echo "Streamlit PID: $STREAMLIT_PID" >> "$LOG_DIR/startup.log"

# Wait for Streamlit to be ready
sleep 15

# Start Serveo tunnel
nohup ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 \
    -R 80:localhost:$PORT serveo.net \
    >> "$LOG_DIR/tunnel.log" 2>&1 &
TUNNEL_PID=$!
echo "Tunnel PID: $TUNNEL_PID" >> "$LOG_DIR/startup.log"

echo "=== Ready ===" >> "$LOG_DIR/startup.log"
echo "URL: https://similarity-search.serveo.net (or check tunnel.log)" >> "$LOG_DIR/startup.log"
