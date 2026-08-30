#!/usr/bin/env bash
set -e

# ==============================================================================
# Local AI Photo Cleaner — 1-Click Bootstrap Runner
# Automatically handles Ollama, models, venv, and starts the cleaner.
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}       Local AI Photo Cleaner (100% Private)          ${NC}"
echo -e "${BLUE}======================================================${NC}\n"

# 1. Check & Install Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}[!] Ollama not found on system.${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${BLUE}[*] Installing Ollama via Homebrew...${NC}"
        brew install ollama
    else
        echo -e "${BLUE}[*] Installing Ollama via official installer...${NC}"
        curl -fsSL https://ollama.com/install.sh | sh
    fi
else
    echo -e "${GREEN}[✓] Ollama installed.${NC}"
fi

# 2. Check if Ollama service is running, start if needed
if ! curl -s http://127.0.0.1:11434/api/tags &> /dev/null; then
    echo -e "${YELLOW}[*] Starting Ollama server in background...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]] && command -v brew &> /dev/null; then
        brew services start ollama 2>/dev/null || (ollama serve &)
    else
        nohup ollama serve > /dev/null 2>&1 &
    fi
    sleep 3
fi
echo -e "${GREEN}[✓] Ollama server is active.${NC}"

# 3. Check & Pull Vision Model
MODEL="llava"
if ! ollama list | grep -q "$MODEL"; then
    echo -e "${YELLOW}[*] Downloading vision model '$MODEL' (first-time only, ~4.5 GB)...${NC}"
    ollama pull "$MODEL"
else
    echo -e "${GREEN}[✓] Vision model '$MODEL' is ready.${NC}"
fi

# 4. Setup Python Virtual Environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}[*] Creating Python virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r requirements.txt
else
    echo -e "${GREEN}[✓] Python environment ready.${NC}"
fi

echo -e "\n${GREEN}[✓] All dependencies configured. Starting cleaner...${NC}\n"

# 5. Run Cleaner with User Arguments
"$VENV_DIR/bin/python" cleaner.py "$@"
