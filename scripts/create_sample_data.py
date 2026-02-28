"""Create minimal sample data for quick demo without full COCO/Flickr30k download."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def download_hf_sample(output_dir: Path, max_images: int = 50):
    """Download small sample from HuggingFace datasets (requires datasets lib)."""
    try:
        from datasets import load_dataset
        import requests
        from PIL import Image
        from io import BytesIO
        import concurrent.futures
        import csv
    except ImportError:
        print("Install: pip install datasets requests Pillow")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading COCO dataset stream...")
    ds = load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split="train", streaming=True)
    
    # We will collect target URLs and Captions first, up to max_images
    tasks = []
    seen_urls = set()
    
    # Curated images
    curated = [
        (
            "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=600&q=80", 
            "Two dogs playing in the snow, running and jumping."
        ),
        (
            "https://images.unsplash.com/photo-1554580665-9831c2063eef?w=600&q=80",
            "A man selling ice cream or desserts from a small cart outdoors."
        ),
        (
            "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=600&q=80",
            "A cat sitting on a windowsill looking out."
        )
    ]
    
    for url, cap in curated:
        tasks.append((url, cap))
        seen_urls.add(url)
        
    for ex in ds:
        if len(tasks) >= max_images:
            break
        url = ex.get("URL")
        cap = ex.get("TEXT")
        if url and cap and url not in seen_urls:
            tasks.append((url, cap))
            seen_urls.add(url)
            
    print(f"Collected {len(tasks)} target URLs. Starting parallel download...")

    ids, paths, captions = [], [], []
    downloaded = 0

    def download_image(index, url, caption):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                fname = f"sample_{index}.jpg"
                out_path = output_dir / fname
                # Resize image slightly to save disk space for large datasets
                img.thumbnail((300, 300))
                img.save(out_path, format="JPEG", quality=85)
                return f"img_{index}", out_path, caption
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_url = {executor.submit(download_image, i, t[0], t[1]): t for i, t in enumerate(tasks)}
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            if result:
                uid, path, cap = result
                ids.append(uid)
                paths.append(path)
                captions.append(cap)
                downloaded += 1
                if downloaded % 100 == 0:
                    print(f"Downloaded {downloaded}/{len(tasks)} images")

    # Write captions to CSV in the parent directory
    csv_path = output_dir.parent / "captions.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for path, cap in zip(paths, captions):
            writer.writerow([path.name, cap])
            
    print(f"Saved {len(ids)} captions to {csv_path}")
    return list(zip(ids, paths, captions))


def create_placeholder_images(output_dir: Path, count: int = 20):
    """Create simple placeholder images (colored squares) for testing."""
    try:
        from PIL import Image
        import random
    except ImportError:
        print("PIL required")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 0, 128)]
    labels = ["red", "green", "blue", "yellow", "purple"]
    for i in range(count):
        idx = i % len(colors)
        img = Image.new("RGB", (224, 224), color=colors[idx])
        fname = f"demo_{i}.jpg"
        path = output_dir / fname
        img.save(path)
        items.append((f"img_{i}", path, labels[idx]))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "images")
    parser.add_argument("--method", choices=["hf", "placeholder"], default="placeholder")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    out = args.output
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if args.method == "placeholder":
        items = create_placeholder_images(out, count=args.count)
    else:
        items = download_hf_sample(out, max_images=args.count)

    print(f"Created {len(items)} images in {out}")
    return items


if __name__ == "__main__":
    main()
