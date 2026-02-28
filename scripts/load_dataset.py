"""Load MS-COCO or Flickr30k dataset for indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def load_coco(data_dir: Path) -> Iterator[tuple[str, Path, str | None]]:
    """
    Load MS-COCO images and captions.
    Yields (id, image_path, caption).
    Expects structure: data_dir/images/, data_dir/annotations/.
    """
    import json

    ann_path = data_dir / "annotations" / "captions_val2017.json"
    img_dir = data_dir / "images" / "val2017"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found at {ann_path}")
    if not img_dir.exists():
        img_dir = data_dir / "val2017"
    if not img_dir.exists():
        raise FileNotFoundError(f"Images dir not found. Tried {data_dir / 'images/val2017'} and {data_dir / 'val2017'}")

    with open(ann_path) as f:
        data = json.load(f)
    img_id_to_file = {x["id"]: x["file_name"] for x in data["images"]}
    captions = {x["image_id"]: [] for x in data["images"]}
    for c in data["annotations"]:
        captions[c["image_id"]].append(c["caption"])

    seen = set()
    for img_id, file_name in img_id_to_file.items():
        path = img_dir / file_name
        if not path.exists():
            continue
        cap = captions.get(img_id, [""])[0] if captions.get(img_id) else None
        uid = f"coco_{img_id}"
        if uid in seen:
            continue
        seen.add(uid)
        yield uid, path, cap


def load_flickr30k(data_dir: Path) -> Iterator[tuple[str, Path, str | None]]:
    """
    Load Flickr30k images and captions.
    Yields (id, image_path, caption).
    Expects: data_dir/flickr30k_images/, data_dir/results.csv or captions.
    """
    import csv

    img_dir = data_dir / "flickr30k_images"
    if not img_dir.exists():
        img_dir = data_dir / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"Images dir not found at {data_dir}")

    csv_path = data_dir / "results.csv"
    if not csv_path.exists():
        csv_path = data_dir / "captions.csv"
    if not csv_path.exists():
        # Fallback: just list images
        for i, p in enumerate(sorted(img_dir.glob("*.jpg"))):
            yield f"flickr_{i}", p, None
        return

    img_to_captions: dict[str, list[str]] = {}
    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 3:
                path = row[0].strip()
                cap = row[2].strip()
                img_to_captions.setdefault(path, []).append(cap)

    for i, (path_key, caps) in enumerate(img_to_captions.items()):
        p = img_dir / path_key
        if not p.exists():
            p = img_dir / Path(path_key).name
        if not p.exists():
            continue
        cap = caps[0] if caps else None
        yield f"flickr_{i}", p, cap


def load_sample_images(data_dir: Path, max_items: int = 100) -> Iterator[tuple[str, Path, str | None]]:
    """
    Load any images from a directory (fallback for quick demo).
    Yields (id, image_path, caption).
    """
    import csv
    
    # Try to load captions mapping if it exists
    captions = {}
    csv_path = data_dir / "captions.csv"
    if not csv_path.exists() and data_dir.parent.exists():
        csv_path = data_dir.parent / "captions.csv"
        
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    captions[row[0].strip()] = row[1].strip()

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    count = 0
    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts and count < max_items:
            cap = captions.get(p.name, None)
            yield f"img_{count}", p, cap
            count += 1

