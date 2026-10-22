#!/usr/bin/env bash
# ==============================================================================
# Local AI Photo Cleaner v3.0 — 1-Click Mac Bootstrap Runner
# Works out of the box on Apple Silicon (M1/M2/M3/M4) & Intel Macs
# ==============================================================================
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BOLD}🌟 Local AI Photo Cleaner v3.0 — Turnkey Mac Photo Curation${NC}"
echo -e "${BLUE}======================================================================${NC}\n"

# 1. Detect Mac Architecture
ARCH=$(uname -m)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        echo -e "${GREEN}[✓] Detected Apple Silicon Mac (${ARCH}) — Metal GPU acceleration enabled.${NC}"
    else
        echo -e "${YELLOW}[!] Detected Intel Mac (${ARCH}) — Running CPU / Ollama mode.${NC}"
    fi
fi

# 2. Setup Python Virtual Environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}[*] Creating isolated Python virtual environment in ${VENV_DIR}...${NC}"
    python3 -m venv "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python3"
PIP_BIN="$VENV_DIR/bin/pip"

# Check dependencies
echo -e "${BLUE}[*] Checking required Python dependencies...${NC}"
$PIP_BIN install --quiet --upgrade pip
$PIP_BIN install --quiet -r requirements.txt

# 3. Handle Source Directory
SOURCE_DIR="$1"

if [ -z "$SOURCE_DIR" ]; then
    echo ""
    echo -e "${YELLOW}Please provide the folder containing your photos and videos.${NC}"
    echo -e "Tip: You can ${BOLD}drag and drop${NC} the folder directly from Finder into this terminal window."
    read -r -p "Enter Photo Folder Path: " INPUT_PATH
    # Strip quotes if drag-and-dropped
    SOURCE_DIR=$(echo "$INPUT_PATH" | sed -e 's/^["'"'"']//' -e 's/["'"'"']$//')
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}[✗] Error: Folder does not exist: '${SOURCE_DIR}'${NC}"
    exit 1
fi

echo -e "\n${GREEN}[✓] Target Folder:${NC} ${SOURCE_DIR}\n"

# 4. Prompt for run mode
echo -e "Choose run mode:"
echo -e "  ${BOLD}1)${NC} Safe Curation (Auto-sort Best Shots, Burst Alternatives, WhatsApp, Docs, Screenshots) ${GREEN}[Default]${NC}"
echo -e "  ${BOLD}2)${NC} Dry Run (Analyze and preview only — do NOT move any files)"
echo -e "  ${BOLD}3)${NC} Ultra-Fast Mode (No AI download — use instant camera focus & burst sharpness)"
read -r -p "Select [1-3] (Default: 1): " MODE_CHOICE

case "$MODE_CHOICE" in
    2)
        echo -e "\n${BLUE}[*] Launching DRY RUN preview...${NC}"
        "$PYTHON_BIN" cleaner.py --source "$SOURCE_DIR" --action dry-run
        ;;
    3)
        echo -e "\n${BLUE}[*] Launching Ultra-Fast Focus Sharpness Mode...${NC}"
        "$PYTHON_BIN" cleaner.py --source "$SOURCE_DIR" --action move --no-ai
        ;;
    *)
        echo -e "\n${BLUE}[*] Launching Safe Curation...${NC}"
        "$PYTHON_BIN" cleaner.py --source "$SOURCE_DIR" --action move
        ;;
esac

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}🎉 All done! Your photos have been curated with 100% privacy.${NC}"
echo -e "${GREEN}======================================================================${NC}"
