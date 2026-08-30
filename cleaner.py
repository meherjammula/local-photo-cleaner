#!/usr/bin/env python3
"""
Local AI Photo Cleaner
Private, 3-tier offline photo sorting tool using Ollama vision models.

Routing Tiers:
1. Main Folder: Confident personal memories (family, kids, selfies, events).
2. _Review_Needed: 50-50 borderline / ambiguous photos (certificates in hand, mixed text/scenes).
3. _Unwanted_Archive: Confirmed junk (screenshots, receipts, catalog products, ads, memes).
"""

import argparse
import io
import os
import shutil
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import ollama
from tqdm import tqdm

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}

def preprocess_image(image_path: Path, max_size: int = 768) -> bytes:
    """Downscale photo to optimize vision inference latency."""
    with Image.open(image_path) as img:
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()

def safe_move(src: Path, dest_dir: Path):
    """Move file with duplicate collision protection."""
    target = dest_dir / src.name
    counter = 1
    while target.exists():
        target = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.move(str(src), str(target))

def evaluate_photo(img_path: Path, client: ollama.Client, model: str) -> str:
    """3-Tier Decision Logic:
    - KEEP (Main folder)
    - REVIEW (_Review_Needed)
    - UNWANTED (_Unwanted_Archive)
    """
    img_bytes = preprocess_image(img_path)

    # Question 1: Is there a real person?
    res1 = client.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": "Is there a real human person in this photo? Answer YES or NO first.",
            "images": [img_bytes]
        }],
        options={"temperature": 0.0},
        keep_alive="10m"
    )
    ans1 = res1["message"]["content"].strip().upper()
    has_person = ans1.startswith("YES") or "YES," in ans1 or "YES " in ans1 or "YES." in ans1

    if not has_person:
        # Check if clear junk vs potential landmark/scenery
        res_junk = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": "Is this a digital screenshot, paper document/receipt, catalog product shot, or a blurry pocket shot? Answer YES or NO.",
                "images": [img_bytes]
            }],
            options={"temperature": 0.0},
            keep_alive="10m"
        )
        ans_junk = res_junk["message"]["content"].strip().upper()
        if ans_junk.startswith("YES") or "YES," in ans_junk or "YES " in ans_junk or "YES." in ans_junk:
            return "UNWANTED"
        return "REVIEW"

    # If real person is present, check if commercial ad graphic or movie scene
    res2 = client.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": "Is this an official commercial advertisement graphic with marketing logos/discounts, or a movie scene with subtitles? Answer YES, NO, or MAYBE.",
            "images": [img_bytes]
        }],
        options={"temperature": 0.0},
        keep_alive="10m"
    )
    ans2 = res2["message"]["content"].strip().upper()
    if ans2.startswith("YES") or "YES," in ans2 or "YES " in ans2:
        return "UNWANTED"
    elif "MAYBE" in ans2 or "UNCERTAIN" in ans2 or "COULD BE" in ans2:
        return "REVIEW"

    return "KEEP"

def process_file(img_path: Path, archive_dir: Path, review_dir: Path, client: ollama.Client, model: str) -> tuple[Path, str]:
    try:
        verdict = evaluate_photo(img_path, client, model)
        if verdict == "UNWANTED":
            safe_move(img_path, archive_dir)
            return img_path, "UNWANTED"
        elif verdict == "REVIEW":
            safe_move(img_path, review_dir)
            return img_path, "REVIEW"
        return img_path, "KEEP"
    except Exception as e:
        return img_path, f"ERROR: {e}"

def main():
    parser = argparse.ArgumentParser(description="Local AI Photo Cleaner — 3-Tier offline photo library sorter.")
    parser.add_argument("--source", required=True, type=str, help="Directory containing photos to sort")
    parser.add_argument("--archive-dir", default="_Unwanted_Archive", type=str, help="Subfolder for confirmed junk")
    parser.add_argument("--review-dir", default="_Review_Needed", type=str, help="Subfolder for 50-50 borderline photos")
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

    archive_path.mkdir(parents=True, exist_ok=True)
    review_path.mkdir(parents=True, exist_ok=True)

    # Scan only top-level files in source (skip the two subfolders)
    images = [
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS and not f.name.startswith(".")
    ]

    total = len(images)
    print(f"📁 Main Photo Library:  {source_path}")
    print(f"📦 Archive Subfolder:   {archive_path}")
    print(f"🔍 Review Subfolder:    {review_path}")
    print(f"🧠 Vision Model:        {args.model}")
    print(f"🖼️  Photos Found:        {total}\n")

    if total == 0:
        print("No photos to process.")
        return

    client = ollama.Client(timeout=args.timeout)

    kept_count = 0
    review_count = 0
    unwanted_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, img, archive_path, review_path, client, args.model): img for img in images}
        with tqdm(total=total, desc="Sorting photos") as pbar:
            for future in as_completed(futures):
                path, result = future.result()
                if result == "KEEP":
                    kept_count += 1
                elif result == "REVIEW":
                    review_count += 1
                elif result == "UNWANTED":
                    unwanted_count += 1
                else:
                    error_count += 1
                pbar.update(1)
                pbar.set_postfix(kept=kept_count, review=review_count, junk=unwanted_count, err=error_count)

    print("\n========== 3-TIER SORTING SUMMARY ==========")
    print(f"Total Processed:                    {total}")
    print(f"✅ Confident Personal Photos (Kept): {kept_count}")
    print(f"🔍 Borderline (Moved to Review):    {review_count}")
    print(f"🗑️  Confirmed Junk (Archived):       {unwanted_count}")
    print(f"⚠️  Errors (Kept in source):         {error_count}")
    print(f"Done.")

if __name__ == "__main__":
    main()
