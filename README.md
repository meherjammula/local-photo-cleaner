# Local AI Photo Cleaner

A 100% private, offline photo sorting tool that separates genuine personal/family memories from WhatsApp clutter (screenshots, receipts, utility bills, memes, e-commerce catalog shots) using local vision models via **Ollama**.

---

## ⚡ The Core Problem It Solves

Standard vision prompts fail on real-world photo dumps because:
1. **The Sunglasses / Candid Trap:** Models often think a child in sunglasses or a casual selfie is a "stock model photo".
2. **The Household Object Trap:** A child holding a branded bag or toy triggers "commercial product" rules.
3. **The Event Decoration Trap:** Stage floral decorations or convocation gowns look like "promotional advertisements".

This tool uses a **2-Stage Hierarchical Decision Pipeline** calibrated to prioritize family life and natural human moments above all else.

---

## 🧠 How the Algorithm Works

```
                     [New Photo]
                          │
                          ▼
            [Stage 1: Real Person Detection]
     "Is there a real human person in this photo?"
                          │
         ┌────────────────┴────────────────┐
         │ NO                              │ YES
         ▼                                 ▼
[UNWANTED ARCHIVE]         [Stage 2: Ad & Meme Filter]
- Digital screenshots       "Is this an official commercial flyer
- Receipts & invoices        or a movie/TV scene with subtitles?"
- Catalog product shots                    │
                            ┌──────────────┴──────────────┐
                            │ YES                         │ NO
                            ▼                             ▼
                   [UNWANTED ARCHIVE]               [KEEP / MAIN]
                   - Marketing ads                  - Family moments
                   - Video memes                    - Kids & babies
                                                    - Candid selfies
                                                    - Travel & dining
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Ollama:** [Download Ollama](https://ollama.com) or `brew install ollama`
- **Vision Model:**
  ```bash
  ollama pull llava
  ```

### 2. Installation
```bash
git clone https://github.com/your-username/local-photo-cleaner.git
cd local-photo-cleaner
pip install -r requirements.txt
```

### 3. Usage
```bash
# Basic run
python cleaner.py --source "/path/to/your/photos"

# Fast parallel run (4 workers on Apple Silicon GPU)
python cleaner.py --source "/path/to/your/photos" --workers 4 --model llava
```

---

## ⚙️ CLI Options

| Flag | Default | Description |
|---|---|---|
| `--source` | *Required* | Path to the photo folder to clean |
| `--archive-dir` | `_Unwanted_Archive` | Name of the folder to move junk into |
| `--model` | `llava` | Local vision model on Ollama (`llava`, `qwen2-vl:7b`, etc.) |
| `--workers` | `4` | Number of concurrent worker threads |
| `--timeout` | `15.0` | Per-photo inference timeout (seconds) |

---

## 🛡️ Safety Guarantees
- **Non-destructive:** Zero photos deleted. All unwanted files are moved safely.
- **Collision Protection:** Automatically renames duplicates (`img_1.jpg`, `img_2.jpg`).
- **100% Offline:** Runs locally via Ollama. No API keys, no internet connection required.
