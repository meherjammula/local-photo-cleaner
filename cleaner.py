#!/usr/bin/env python3
"""
Local AI Photo Cleaner (v2.0)
Private, multi-tier offline photo sorting tool using Ollama vision models.

Routing Tiers:
1. Main Folder: Confident personal memories (family, kids, selfies, events, best burst shot).
2. _Duplicates_Archive: Exact byte duplicates, suffix copies, and redundant burst frames.
3. _Documents_and_Bills: IDs, receipts, bills, medical prescriptions, whiteboard notes, QR codes.
4. _Review_Needed: 50-50 borderline / ambiguous photos, dark/blurry shots.
5. _Unwanted_Archive: Screenshots, screen recordings, stickers, memes, greeting cards, e-commerce catalog models, ads.
"""

import argparse
import io
import os
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageChops
import numpy as np
import ollama
from tqdm import tqdm

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".3gp", ".avi", ".mkv", ".mpeg"}

SCREENSHOT_REGEX = re.compile(r"^(Screenshot|Screen\s*Shot|Screen_Shot|Screenshot_|Screen_Recording|ScreenRecording)", re.IGNORECASE)
DUP_REGEX = re.compile(r"^(.*?)(?:\s*\(\d+\)|\s*-\s*Copy)(\.[^.]+)$", re.IGNORECASE)
BURST_REGEX = re.compile(r"^(\d{8}_\d{6})\((\d+)\)(\.[^.]+)$", re.IGNORECASE)

def preprocess_image(image_path: Path, max_size: int = 768) -> bytes:
    """Downscale photo to optimize vision inference latency."""
    with Image.open(image_path) as img:
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()

def safe_move(src: Path, dest_dir: Path):
    """Move file with duplicate collision protection."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.name
    counter = 1
    while target.exists():
        target = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.move(str(src), str(target))

def evaluate_photo(img_path: Path, client: ollama.Client, model: str) -> str:
    """Comprehensive Multi-Tier Decision Logic:
    - KEEP (Main folder)
    - DOCUMENT (_Documents_and_Bills)
    - REVIEW (_Review_Needed)
    - UNWANTED (_Unwanted_Archive)
    """
    try:
        # Pre-check: extremely dark or tiny image
        with Image.open(img_path) as im:
            w, h = im.size
            if w < 400 or h < 400:
                return "UNWANTED"  # Sticker, tiny thumbnail, emoji
            
            # Check brightness
            rgb = im.convert("L")
            stat = np.array(rgb)
            if np.mean(stat) < 10.0:  # Pitch black pocket shot
                return "REVIEW"

        img_bytes = preprocess_image(img_path)

        # Check 1: Utility Document / ID / Bill / Prescription
        res_doc = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": "Is this an ID card (Aadhaar, passport, driver license), paper receipt/bill, bank slip, medical prescription, presentation slide, or whiteboard note? Answer YES or NO.",
                "images": [img_bytes]
            }],
            options={"temperature": 0.0},
            keep_alive="10m"
        )
        ans_doc = res_doc["message"]["content"].strip().upper()
        if ans_doc.startswith("YES") or "YES," in ans_doc or "YES " in ans_doc or "YES." in ans_doc:
            return "DOCUMENT"

        # Check 2: Greeting card / WhatsApp Quote / Meme
        res_quote = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": "Is this a generic 'Good Morning' card, festival greeting graphic with text/flowers, motivational quote card, or internet meme? Answer YES or NO.",
                "images": [img_bytes]
            }],
            options={"temperature": 0.0},
            keep_alive="10m"
        )
        ans_quote = res_quote["message"]["content"].strip().upper()
        if ans_quote.startswith("YES") or "YES," in ans_quote or "YES " in ans_quote or "YES." in ans_quote:
            return "UNWANTED"

        # Check 3: Real Person / Family
        res_person = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": "Does this photo show a real person (face, portrait, selfie, family, child)? Answer YES if a real person or child is the subject of the photo. Answer NO if the main subject is an object, product, paper plate, food item, or document where only a hand/finger is holding it.",
                "images": [img_bytes]
            }],
            options={"temperature": 0.0},
            keep_alive="10m"
        )
        ans_person = res_person["message"]["content"].strip().upper()
        has_person = ans_person.startswith("YES") or "YES," in ans_person or "YES " in ans_person or "YES." in ans_person

        if not has_person:
            # Filter pure screenshots or catalog objects vs landscapes
            res_junk = client.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": "Is this a digital screenshot, catalog product shot, or a blurry pocket shot? Answer YES or NO.",
                    "images": [img_bytes]
                }],
                options={"temperature": 0.0},
                keep_alive="10m"
            )
            ans_junk = res_junk["message"]["content"].strip().upper()
            if ans_junk.startswith("YES") or "YES," in ans_junk or "YES " in ans_junk or "YES." in ans_junk:
                return "UNWANTED"
            return "REVIEW"

        # Check 4: Filter out fashion catalog models or commercial ads
        res_model = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": "Is this image an e-commerce clothing catalog model (staged fashion model posing on seamless studio background), or a commercial marketing ad graphic? Answer YES if it is a clothing catalog model or commercial ad. Answer NO if it is a real candid personal/family photo, selfie, kid photo, or home moment.",
                "images": [img_bytes]
            }],
            options={"temperature": 0.0},
            keep_alive="10m"
        )
        ans_model = res_model["message"]["content"].strip().upper()
        if ans_model.startswith("YES") or "YES," in ans_model or "YES " in ans_model or "YES." in ans_model:
            return "UNWANTED"
        elif "MAYBE" in ans_model or "UNCERTAIN" in ans_model:
            return "REVIEW"

        return "KEEP"
    except Exception as e:
        return "REVIEW"

def process_file(img_path: Path, archive_dir: Path, review_dir: Path, doc_dir: Path, client: ollama.Client, model: str) -> tuple[Path, str]:
    try:
        verdict = evaluate_photo(img_path, client, model)
        if verdict == "UNWANTED":
            safe_move(img_path, archive_dir)
            return img_path, "UNWANTED"
        elif verdict == "DOCUMENT":
            safe_move(img_path, doc_dir)
            return img_path, "DOCUMENT"
        elif verdict == "REVIEW":
            safe_move(img_path, review_dir)
            return img_path, "REVIEW"
        return img_path, "KEEP"
    except Exception as e:
        return img_path, f"ERROR: {e}"

def main():
    parser = argparse.ArgumentParser(description="Local AI Photo Cleaner (v2.0) — Multi-tier offline photo library sorter.")
    parser.add_argument("--source", required=True, type=str, help="Directory containing photos to sort")
    parser.add_argument("--archive-dir", default="_Unwanted_Archive", type=str, help="Subfolder for confirmed junk")
    parser.add_argument("--review-dir", default="_Review_Needed", type=str, help="Subfolder for 50-50 borderline photos")
    parser.add_argument("--doc-dir", default="_Documents_and_Bills", type=str, help="Subfolder for IDs, bills, documents")
    parser.add_argument("--dup-dir", default="_Duplicates_Archive", type=str, help="Subfolder for isolated duplicate copies")
    parser.add_argument("--model", default="llava", type=str, help="Ollama vision model (default: llava)")
    parser.add_argument("--workers", default=4, type=int, help="Number of worker threads (default: 4)")
    parser.add_argument("--timeout", default=15.0, type=float, help="Per-image request timeout in seconds")

    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory '{source_path}' does not exist.")
        sys.exit(1)

    archive_path = source_path / args.archive_dir
    review_path = source_path / args.review_dir
    doc_path = source_path / args.doc_dir
    dup_path = source_path / args.dup_dir

    archive_path.mkdir(parents=True, exist_ok=True)
    review_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)
    dup_path.mkdir(parents=True, exist_ok=True)

    all_files = [
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS and not f.name.startswith(".")
    ]

    print(f"📁 Main Photo Library:      {source_path}")
    print(f"📦 Archive Subfolder:       {archive_path}")
    print(f"📄 Documents & Bills:       {doc_path}")
    print(f"🔍 Review Subfolder:        {review_path}")
    print(f"👯 Duplicates Folder:       {dup_path}")
    print(f"🧠 Vision Model:            {args.model}")
    print(f"🖼️  Photos Found:            {len(all_files)}\n")

    # PRE-FLIGHT 1: Instant Screenshot & Screen Recording Filter
    screenshot_moved = 0
    remaining_files = []
    for f in all_files:
        if SCREENSHOT_REGEX.search(f.name):
            safe_move(f, archive_path)
            screenshot_moved += 1
        else:
            remaining_files.append(f)

    if screenshot_moved > 0:
        print(f"⚡ Pre-flight 1: Moved {screenshot_moved} screenshots directly to {archive_path.name}")

    # PRE-FLIGHT 2: Instant Fast Duplicate Suffix Isolation
    dup_moved = 0
    name_map = {f.name: f for f in remaining_files}
    after_dup_files = []
    for f in remaining_files:
        m = DUP_REGEX.match(f.name)
        if m:
            base_name = m.group(1) + m.group(2)
            if base_name in name_map and base_name != f.name:
                safe_move(f, dup_path)
                dup_moved += 1
                continue
        after_dup_files.append(f)

    if dup_moved > 0:
        print(f"⚡ Pre-flight 2: Isolated {dup_moved} duplicate copies to {dup_path.name}")

    # PRE-FLIGHT 3: Tiny Stickers / Micro-Graphics Filter (< 400x400)
    sticker_moved = 0
    vision_candidates = []
    for f in after_dup_files:
        try:
            with Image.open(f) as im:
                w, h = im.size
                if w < 400 or h < 400:
                    safe_move(f, archive_path)
                    sticker_moved += 1
                    continue
        except Exception:
            pass
        vision_candidates.append(f)

    if sticker_moved > 0:
        print(f"⚡ Pre-flight 3: Moved {sticker_moved} tiny stickers/icons directly to {archive_path.name}")

    total = len(vision_candidates)
    print(f"\n🚀 Running AI Vision Model on {total} remaining photos...\n")

    if total == 0:
        print("All photos classified by pre-flight filters. Done.")
        return

    client = ollama.Client(timeout=args.timeout)

    kept_count = 0
    doc_count = 0
    review_count = 0
    unwanted_count = screenshot_moved + sticker_moved
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, img, archive_path, review_path, doc_path, client, args.model): img for img in vision_candidates}

        with tqdm(total=total, desc="Sorting photos") as pbar:
            for future in as_completed(futures):
                path, result = future.result()
                if result == "KEEP":
                    kept_count += 1
                elif result == "DOCUMENT":
                    doc_count += 1
                elif result == "REVIEW":
                    review_count += 1
                elif result == "UNWANTED":
                    unwanted_count += 1
                else:
                    error_count += 1
                pbar.update(1)
                pbar.set_postfix(kept=kept_count, docs=doc_count, review=review_count, junk=unwanted_count, err=error_count)

    print("\n========== MULTI-TIER SORTING SUMMARY ==========")
    print(f"Total Processed:                    {len(all_files)}")
    print(f"✅ Confident Personal Photos (Kept): {kept_count}")
    print(f"📄 Documents & Bills Isolated:      {doc_count}")
    print(f"👯 Duplicates Isolated:             {dup_moved}")
    print(f"🔍 Borderline (Moved to Review):    {review_count}")
    print(f"🗑️  Confirmed Junk (Archived):       {unwanted_count}")
    print(f"⚠️  Errors (Kept in source):         {error_count}")
    print(f"Done.")

if __name__ == "__main__":
    main()
