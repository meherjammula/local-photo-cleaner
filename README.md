# 🌟 Local AI Photo Cleaner (v3.0)

> **Private, Offline Photo & Video Curation Engine with Smart Burst "Best-Shot" Selection.**  
> Powered by Apple Silicon Metal GPU acceleration (`mlx-vlm`) & local vision intelligence.

---

## 💡 Why This Tool?

When you dump years of photos from old phones, Google Photos, or backups (250–500 GB), you face two massive headaches:

1. **The Burst / Series Dilemma**: You took 5 shots of someone at an event. In one shot, they are looking directly into the camera lens with open eyes and a natural smile—that's the **best shot**. The other 4 shots are redundant, blurry, or captured mid-blink.
2. **The Junk Flood**: Thousands of WhatsApp forwards, screenshots, receipt bills, dark pocket shots, memes, and stickers cluttering your genuine family memories.

**Local AI Photo Cleaner** automatically runs directly on your Mac, analyzes your photos completely offline, picks the **single best shot** from every burst, and cleanly organizes the clutter into safe subfolders—**without deleting a single file**.

---

## ⚡ 1-Click Quickstart (for Mac)

Open your **Terminal** app on your Mac and run:

```bash
# 1. Clone this repository
git clone https://github.com/meherjammula/local-photo-cleaner.git
cd local-photo-cleaner

# 2. Run the 1-click setup
./run.sh
```

When prompted, simply **drag and drop your photo folder** from Finder directly into the terminal window and press **Enter**!

---

## 🗂️ How Your Photos Get Organized

Your original folder is kept clean and curated with only your **genuine best memories**. All clutter is separated into dedicated, easily reviewable subfolders:

```
Your_Photos_Folder/
│
├── 🌟 [Camera Originals / Best Shots]   <-- Kept in main folder! Only the sharpest, best-expression shots
│
├── 📁 _BURST_ALTERNATIVES/              <-- Redundant burst takes & closed-eye alternative shots
├── 📁 _WHATSAPP_JUNK/                   <-- WhatsApp images, forwarded videos, greeting cards
├── 📁 _SCREENSHOTS/                     <-- Mobile and desktop display captures
├── 📁 _DOCUMENTS_AND_BILLS/             <-- Receipts, bills, invoices, ID cards, prescriptions
├── 📁 _STICKERS_TINY/                   <-- Micro-graphics, stickers, icons (< 400x400 px)
├── 📁 _DARK_POCKET_SHOTS/               <-- Accidental pitch-black pocket or purse shots
├── 📁 _DUPLICATES_EXACT/                <-- Identical byte-for-byte duplicate copies
├── 📁 _UNWANTED_CLUTTER/                <-- E-commerce clothing catalogs, internet memes, flyers
│
└── 🔄 undo_cleaner.py                   <-- Automatic 1-click rollback script (revert anytime!)
```

---

## 🎯 Key Features

### 1. Smart Burst & Series "Best Shot" Selector
- Groups photos taken in quick succession ($\le 6$ seconds) of the same subject.
- Measures optical focus using Laplacian edge variance ($Var(\nabla^2 I)$).
- Evaluates **eye contact, open eyes (no blinks), and facial expression** using local vision AI.
- Crown the **best shot** for your primary album, moving secondary takes to `_BURST_ALTERNATIVES/`.

### 2. 100% Privacy & Offline (No Cloud Uploads)
- Everything runs 100% locally on your machine.
- Zero network calls. Your family photos and documents never touch any remote server or cloud.

### 3. Industrial Scale (250 GB – 500 GB) & Live Monitoring
- **SQLite WAL Checkpoint Engine**: If your laptop sleeps, runs out of battery, or you close the app, running it again **resumes instantly** without re-scanning already processed files.
- **Mac Desktop Notifications**: Sends native macOS notification banners at key milestones (25%, 50%, 75%, 100%) and every 15 minutes with speed & time remaining.
- **Live `progress_summary.txt` in Finder**: You don't have to keep staring at the terminal. You can simply look inside your photo folder in Finder to see real-time scanned counts, speeds, and ETA.
- **Completion Chime**: Plays a gentle Mac chime when all 500 GB are fully curated.

### 4. 100% Non-Destructive with 1-Click Rollback
- No files are ever permanently deleted.
- Every run automatically creates an `undo_cleaner.py` script. Running `python3 undo_cleaner.py` moves every single file back to its exact original place.

---

## 🧠 AI Vision Models & Hardware Scaling Guide

By default, the tool runs **`mlx-community/Qwen2.5-VL-3B-Instruct-4bit`**, which is optimized for high-speed local inference across all Apple Silicon Macs.

If you have a high-spec Mac (16 GB, 24 GB, 36 GB, 48 GB, or 64 GB Unified Memory), you can switch models anytime:

| Model | Parameters | 4-bit Size | Recommended Mac RAM | Command Example |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL-3B** *(Default)* | **3 Billion** | **~2.9 GB** | **8 GB – 16 GB** (Air / Pro) | `./run.sh` *(Default)* |
| **Qwen2.5-VL-7B** *(Pro / Studio)* | **7 Billion** | **~5.2 GB** | **16 GB – 48 GB+** (Pro / Max) | `python3 cleaner.py --source /path --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit` |
| **Qwen2.5-VL-72B** *(Ultra Studio)* | **72 Billion** | **~42 GB** | **64 GB – 128 GB+** (Max / Ultra) | `python3 cleaner.py --source /path --model mlx-community/Qwen2.5-VL-72B-Instruct-4bit` |

### 💡 Why Qwen2.5-VL vs. Llama 3.2 Vision?
While Llama 3.2 Vision (11B) is also open, **Qwen2.5-VL** is specifically superior for photo curation because:
1. **Dynamic Native Resolution**: Qwen processes photos at their true aspect ratio without arbitrarily slicing them into square tiles.
2. **Fine Facial Nuance & OCR**: Qwen2.5-VL-7B significantly outperforms Llama 3.2 in detecting subtle micro-blinks, direct eye-contact angles, and fine document receipts.
3. **Optimized Apple Silicon MLX Kernels**: Runs natively on Apple Metal GPU at 60–100+ tokens/second.

---

## 🛠️ Advanced CLI Usage

If you prefer using the command line directly:

```bash
# Basic run (moves files into organized subfolders)
python3 cleaner.py --source "/path/to/photos"

# Dry Run (Preview counts without moving any files)
python3 cleaner.py --source "/path/to/photos" --action dry-run

# Copy instead of move (keeps original folder completely untouched)
python3 cleaner.py --source "/path/to/photos" --dest "/path/to/curated_output" --action copy

# Ultra-fast mode (uses camera focus & burst sharpness without AI download)
python3 cleaner.py --source "/path/to/photos" --no-ai

# Custom burst capture time window (e.g., 10 seconds)
python3 cleaner.py --source "/path/to/photos" --burst-window 10.0
```

---

## 💻 System Requirements
- **macOS** (Optimized for Apple Silicon M1, M2, M3, M4 Macs with native Metal GPU acceleration).
- **Python 3.10+**.
- Also compatible with Intel Macs and Linux.

---

## 🤝 Feedback & Support
Have ideas, questions, or issues? Open an Issue or reach out—you are our valued alpha tester!
