#!/usr/bin/env python3
"""
Local AI Photo Cleaner
Private, offline WhatsApp & photo dump cleaner using Ollama vision models.
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
    """Downscale photo to reduce vision model inference latency."""
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
    """2-Stage Hierarchical Classification:
    Stage 1: Person detection (High recall)
    Stage 2: Ad / Meme detection (High precision)
    """
    img_bytes = preprocess_image(img_path)

    # Stage 1: Is there a real human?
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
        return "JUNK_NON_HUMAN"

    # Stage 2: Verify not a commercial ad graphic or movie subtitle
    res2 = client.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": "Is this an official commercial advertisement graphic with marketing logos/discounts, or a movie scene with subtitles? Answer YES or NO first.",
            "images": [img_bytes]
        }],
        options={"temperature": 0.0},
        keep_alive="10m"
    )
    ans2 = res2["message"]["content"].strip().upper()
    is_ad_or_movie = ans2.startswith("YES") or "YES," in ans2 or "YES " in ans2 or "YES." in ans2

    if is_ad_or_movie:
        return "JUNK_AD_OR_MOVIE"

    return "KEEP"

def process_file(img_path: Path, archive_dir: Path, client: ollama.Client, model: str) -> tuple[Path, str]:
    try:
        verdict = evaluate_photo(img_path, client, model)
        if verdict.startswith("JUNK"):
            safe_move(img_path, archive_dir)
            return img_path, verdict
        return img_path, "KEEP"
    except Exception as e:
        return img_path, f"ERROR: {e}"

def main():
    parser = argparse.ArgumentParser(description="Local AI Photo Cleaner — Sort personal photos from junk offline.")
    parser.add_argument("--source", required=True, type=str, help="Directory containing photos to sort")
    parser.add_argument("--archive-dir", default="_Unwanted_Archive", type=str, help="Name of subfolder for junk files")
    parser.add_argument("--model", default="llava", type=str, help="Ollama vision model (default: llava)")
    parser.add_argument("--workers", default=4, type=int, help="Number of worker threads (default: 4)")
    parser.add_argument("--timeout", default=15.0, type=float, help="Per-image request timeout in seconds")

    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory '{source_path}' does not exist.")
        sys.exit(1)

    archive_path = source_path / args.archive_dir
    archive_path.mkdir(parents=True, exist_ok=True)

    # Gather valid image files directly in source (skip archive directory)
    images = [
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS and not f.name.startswith(".")
    ]

    total = len(images)
    print(f"📁 Source Folder:   {source_path}")
    print(f"📦 Archive Folder:  {archive_path}")
    print(f"🧠 Vision Model:    {args.model}")
    print(f"🖼️  Photos Found:    {total}\n")

    if total == 0:
        print("No photos to process.")
        return

    client = ollama.Client(timeout=args.timeout)

    kept_count = 0
    non_human_count = 0
    ad_movie_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, img, archive_path, client, args.model): img for img in images}
        with tqdm(total=total, desc="Cleaning photos") as pbar:
            for future in as_completed(futures):
                path, result = future.result()
                if result == "KEEP":
                    kept_count += 1
                elif result == "JUNK_NON_HUMAN":
                    non_human_count += 1
                elif result == "JUNK_AD_OR_MOVIE":
                    ad_movie_count += 1
                else:
                    error_count += 1
                pbar.update(1)
                pbar.set_postfix(kept=kept_count, junk=non_human_count + ad_movie_count, errors=error_count)

    print("\n========== SUMMARY ==========")
    print(f"Total Processed:         {total}")
    print(f"Personal Photos Kept:    {kept_count}")
    print(f"Junk Archived:           {non_human_count + ad_movie_count} (Non-human: {non_human_count}, Ads/Memes: {ad_movie_count})")
    print(f"Errors (Kept in source): {error_count}")
    print(f"Done.")

if __name__ == "__main__":
    main()
