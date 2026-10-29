"""Load only reviewed project labels, preserving image aspect ratio."""

import hashlib
import json
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from src.annotations import validate_boxes
from src.prepare_data import ROOT


def letterbox(image, boxes):
    width, height = image.size
    scale = min(320 / width, 320 / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    left, top = (320 - resized_width) // 2, (320 - resized_height) // 2
    canvas = Image.new("RGB", (320, 320), (114, 114, 114))
    canvas.paste(image.resize((resized_width, resized_height)), (left, top))
    transformed = boxes.clone()
    transformed[:, [0, 2]] = (transformed[:, [0, 2]] * resized_width / width + left) / 320
    transformed[:, [1, 3]] = (transformed[:, [1, 3]] * resized_height / height + top) / 320
    return canvas, transformed


def horizontal_flip(image, boxes):
    flipped = boxes.clone()
    flipped[:, 0] = image.width - boxes[:, 2]
    flipped[:, 2] = image.width - boxes[:, 0]
    return ImageOps.mirror(image), flipped


class VehicleDataset(Dataset):
    def __init__(self, split, augment=False):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown dataset split")
        if augment and split != "train":
            raise ValueError("Augmentation is training-only")
        self.augment = augment
        self.document = json.loads((ROOT / f"curated_dataset/annotations/{split}.json").read_text())
        self.images = self.document["images"]
        incomplete = [image["source_id"] for image in self.images if image["qa_status"] != "reviewed"]
        if incomplete:
            raise ValueError(f"{split}: {len(incomplete)} unreviewed images; do not train/evaluate incomplete labels")
        self.boxes = {image["id"]: [] for image in self.images}
        if len(self.boxes) != len(self.images) or not self.images:
            raise ValueError("Empty dataset or duplicate image IDs")
        for annotation in self.document["annotations"]:
            if annotation["category_id"] != 1 or annotation["image_id"] not in self.boxes:
                raise ValueError("Unknown category or image ID")
            left, top, width, height = annotation["bbox"]
            self.boxes[annotation["image_id"]].append([left, top, left + width, top + height])
        # Verify the exact data once, before optimization starts.
        for image in self.images:
            path = (ROOT / image["file_name"]).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("Image path is outside project")
            if hashlib.sha256(path.read_bytes()).hexdigest() != image["sha256"]:
                raise ValueError(f"Image hash mismatch: {image['source_id']}")
            validate_boxes(self.boxes[image["id"]], image["width"], image["height"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        record = self.images[index]
        with Image.open(ROOT / record["file_name"]) as source:
            image = source.convert("RGB")
        boxes = torch.tensor(self.boxes[record["id"]], dtype=torch.float32).reshape(-1, 4)
        if self.augment:
            if random.random() < 0.5:
                image, boxes = horizontal_flip(image, boxes)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        image, boxes = letterbox(image, boxes)
        array = np.asarray(image, dtype=np.float32).copy() / 255
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, boxes, record["source_id"]


def collate_batch(batch):
    images, boxes, identifiers = zip(*batch)
    return torch.stack(images), list(boxes), list(identifiers)