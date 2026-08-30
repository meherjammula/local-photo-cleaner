# Local AI Photo Cleaner

A 100% private, offline photo sorting tool that categorizes photo dumps into **3 clean tiers** using local vision models via **Ollama**.

---

## 🎯 3-Tier Destination Routing

When given a photo directory, it organizes your files into 3 distinct places:

| Tier / Location | What goes here | Examples |
|---|---|---|
| **`Main Folder`** | **Confident Personal Memories** | Family life, kids, selfies, dining, vacations, pets, celebrations |
| **`_Review_Needed/`** | **50-50 Borderline Photos** | Certificates in hand, mixed text/scenes, ambiguous candidate forwards |
| **`_Unwanted_Archive/`** | **Confirmed Junk** | Digital screenshots, bills, receipts, furniture catalog shots, ads, memes |

---

## 🧠 Decision Flowchart

```
                             [Photo]
                                │
                                ▼
                  [1. Is there a real person?]
                                │
               ┌────────────────┴────────────────┐
               │ NO                              │ YES
               ▼                                 ▼
   [2. Clear document/catalog?]      [3. Commercial ad or movie meme?]
         ┌─────┴─────┐                     ┌─────┴─────┐
     YES │       NO  │                 YES │       NO  │
         ▼           ▼                     ▼           ▼
   [_Unwanted]   [_Review]           [_Unwanted]    [KEEP / MAIN]
   (Receipts)   (Landscapes)         (Ads/Memes)   (Family/Life)
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
| `--review-dir` | `_Review_Needed` | Name of the folder for 50-50 borderline photos |
| `--archive-dir` | `_Unwanted_Archive` | Name of the folder for confirmed junk files |
| `--model` | `llava` | Local vision model on Ollama (`llava`, `qwen2-vl:7b`, etc.) |
| `--workers` | `4` | Number of concurrent worker threads |
| `--timeout` | `15.0` | Per-photo inference timeout (seconds) |

---

## 🛡️ Safety Guarantees
- **Non-destructive:** Zero photos deleted. All moves are safely isolated.
- **Collision Protection:** Automatically renames duplicates (`img_1.jpg`, `img_2.jpg`).
- **100% Offline:** Runs locally via Ollama. No API keys, no cloud upload.
