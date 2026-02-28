"""CLIP-based encoder for images and text."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
import open_clip

from app.config import settings


class EncoderService:
    """Encode images and text to shared CLIP embedding space with L2 normalization."""

    DIMENSION = 512  # ViT-B-32 output dimension

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the CLIP model."""
        if self._model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                settings.clip_model_name,
                pretrained=settings.clip_pretrained,
            )
            tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to L2-normalized embedding."""
        self._ensure_loaded()
        with torch.no_grad():
            tokens = self._tokenizer([text])
            text_features = self._model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.numpy().astype(np.float32).squeeze(0)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Batch encode texts to L2-normalized embeddings."""
        if not texts:
            return np.array([]).reshape(0, self.DIMENSION)
        self._ensure_loaded()
        with torch.no_grad():
            tokens = self._tokenizer(texts)
            text_features = self._model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.numpy().astype(np.float32)

    def encode_image(self, image: Union[Image.Image, str, Path]) -> np.ndarray:
        """Encode single image to L2-normalized embedding."""
        self._ensure_loaded()
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be PIL Image, path string, or Path")
        img_tensor = self._preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_features = self._model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.numpy().astype(np.float32).squeeze(0)

    def encode_images(
        self, images: list[Union[Image.Image, str, Path]], batch_size: int = 32
    ) -> np.ndarray:
        """Batch encode images to L2-normalized embeddings."""
        if not images:
            return np.array([]).reshape(0, self.DIMENSION)
        self._ensure_loaded()
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert("RGB")
                tensors.append(self._preprocess(img))
            stack = torch.stack(tensors)
            with torch.no_grad():
                feats = self._model.encode_image(stack)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.numpy().astype(np.float32))
        return np.vstack(embeddings)
