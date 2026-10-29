"""Render images for visual labeling and validate our own annotation records."""

import argparse
import csv
import json
from collections import Counter

from PIL import Image, ImageDraw

from src.prepare_data import ROOT


LABEL_PATH = ROOT / "data/vehicle_boxes.json"


def manifest():
    with (ROOT / "data/manifest.csv").open() as handle:
        return list(csv.DictReader(handle))


def read_labels():
    if not LABEL_PATH.exists():
        return {}
    return json.loads(LABEL_PATH.read_text())["images"]


def validate_boxes(boxes, width, height):
    for box in boxes:
        if len(box) != 4:
            raise ValueError(f"Expected x1,y1,x2,y2, got {box}")
        left, top, right, bottom = box
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"Out-of-bounds or inverted box: {box}")
        if right - left < 20 or bottom - top < 20:
            raise ValueError(f"Box below the original-image 20x20 minimum: {box}")


def render(start, count, pilot=False, overlays=False):
    rows = manifest()
    if pilot:
        rows = [row for row in rows if row["pilot"] == "1"]
    labels = read_labels()
    output = ROOT / "reports/figures/annotation"
    output.mkdir(parents=True, exist_ok=True)
    for row in rows[start:start + count]:
        with Image.open(ROOT / row["image_path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        if overlays:
            for number, box in enumerate(labels.get(row["image_id"], {}).get("boxes", []), 1):
                draw.rectangle(box, outline="lime", width=2)
                draw.text((box[0], max(0, box[1] - 12)), str(number), fill="lime")
        else:
            # Coordinate guides assist visual labeling, not object discovery.
            for horizontal in range(0, image.width, 100):
                draw.line((horizontal, 0, horizontal, image.height), fill=(90, 90, 90), width=1)
                draw.text((horizontal + 2, 8), str(horizontal), fill="yellow")
            for vertical in range(0, image.height, 100):
                draw.line((0, vertical, image.width, vertical), fill=(90, 90, 90), width=1)
                draw.text((2, vertical + 2), str(vertical), fill="yellow")
        name = f"{row['image_id']}_{'review' if overlays else 'raw'}.jpg"
        image.save(output / name, quality=95)
        print(f"{row['image_id']} {row['split']} {output / name}")


def export(require_complete=False):
    rows = manifest()
    labels = read_labels()
    known = {row["image_id"] for row in rows}
    if set(labels) - known:
        raise ValueError("Label records contain unknown image IDs")
    counts = Counter()
    for row in rows:
        record = labels.get(row["image_id"])
        if record is None:
            row["annotation_status"] = "pending"
            row["qa_status"] = "pending"
            continue
        validate_boxes(record["boxes"], int(row["width"]), int(row["height"]))
        if record["status"] not in ("draft", "reviewed"):
            raise ValueError("Status must be draft or reviewed")
        row["annotation_status"] = "complete" if record["status"] == "reviewed" else "draft"
        row["qa_status"] = record["status"]
        counts[record["status"]] += 1
    pending = sum(row["qa_status"] != "reviewed" for row in rows)
    if require_complete and pending:
        raise ValueError(f"Dataset not ready: {pending} images lack reviewed labels")
    output = ROOT / "curated_dataset/annotations"
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        images, annotations = [], []
        for row in rows:
            if row["split"] != split:
                continue
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": row["image_path"],
                           "width": int(row["width"]), "height": int(row["height"]),
                           "source_id": row["image_id"], "video_id": row["video_id"],
                           "sha256": row["sha256"], "qa_status": row["qa_status"]})
            for box in labels.get(row["image_id"], {}).get("boxes", []):
                left, top, right, bottom = box
                width, height = right - left, bottom - top
                annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                    "category_id": 1, "bbox": [left, top, width, height],
                                    "area": width * height, "iscrowd": 0})
        document = {"info": {"annotation_source": "AI visual inspection; no supplied labels",
                             "independent_human_audit": False},
                    "images": images, "annotations": annotations,
                    "categories": [{"id": 1, "name": "vehicle"}]}
        (output / f"{split}.json").write_text(json.dumps(document, indent=2) + "\n")
    with (ROOT / "data/manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Labels: {dict(counts)}; pending or unreviewed: {pending}/500")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["render", "export"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--overlays", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "render":
        render(arguments.start, arguments.count, arguments.pilot, arguments.overlays)
    else:
        export(arguments.require_complete)