"""Batch index images into the vector database."""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.services.encoder import EncoderService
from app.services.retriever import RetrieverService
from scripts.load_dataset import load_coco, load_flickr30k, load_sample_images


def main():
    parser = argparse.ArgumentParser(description="Index images into vector DB")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Data directory")
    parser.add_argument("--dataset", choices=["coco", "flickr30k", "sample"], default="sample")
    parser.add_argument("--max-items", type=int, default=1000, help="Max images to index (sample mode)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    data_dir = data_dir.resolve()

    if args.dataset == "coco":
        items = list(load_coco(data_dir))
    elif args.dataset == "flickr30k":
        items = list(load_flickr30k(data_dir))
    else:
        img_dir = data_dir / "images"
        if not img_dir.exists():
            img_dir = data_dir
        items = list(load_sample_images(img_dir, max_items=args.max_items))

    if not items:
        print("No images found. Ensure data directory structure is correct.")
        print("  - COCO: data_dir/annotations/captions_val2017.json, data_dir/images/val2017/")
        print("  - Flickr30k: data_dir/flickr30k_images/, data_dir/results.csv")
        print("  - Sample: data_dir/images/*.jpg (or data_dir/*.jpg)")
        sys.exit(1)

    print(f"Indexing {len(items)} images...")
    encoder = EncoderService()
    retriever = RetrieverService(encoder=encoder)

    ids = []
    paths = []
    captions = []
    for uid, path, cap in items:
        ids.append(uid)
        paths.append(path)
        captions.append(cap)

    embeddings = encoder.encode_images(paths, batch_size=args.batch_size)
    metadatas = [
        {"path": str(p), "caption": c or ""}
        for p, c in zip(paths, captions)
    ]
    retriever.add(ids, embeddings, metadatas)
    print(f"Indexed {len(ids)} images. Total in index: {retriever.count()}")


if __name__ == "__main__":
    main()
