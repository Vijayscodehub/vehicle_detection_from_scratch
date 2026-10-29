"""Inventory and preview raw frames without reading supplied annotations."""

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset/images20-track-train-1/bdd100k/images/track/train"


def spaced_indices(count, requested):
    """Include both ends, with no repeated frame positions."""
    if requested < 2 or count < requested:
        raise ValueError("Need at least two samples and enough source frames")
    return [round(index * (count - 1) / (requested - 1))
            for index in range(requested)]


def inventory():
    sequences = {}
    for folder in sorted(RAW.iterdir()):
        if folder.is_dir():
            frames = sorted(folder.glob("*.jpg"))
            if frames:
                sequences[folder.name] = frames
    output = ROOT / "data/source_inventory.csv"
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "frame_count", "first_frame", "last_frame"])
        for video_id, frames in sequences.items():
            writer.writerow([video_id, len(frames), frames[0].name, frames[-1].name])
    return sequences


def previews(sequences):
    candidates = sorted(sequences)
    random.Random(42).shuffle(candidates)
    candidates = candidates[:60]
    output = ROOT / "reports/figures/screening"
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/screening_frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "video_id", "source_path", "sha256"])
        for batch in range(12):
            sheet = Image.new("RGB", (1536, 5 * 320), "white")
            draw = ImageDraw.Draw(sheet)
            for row, video_id in enumerate(candidates[batch * 5:(batch + 1) * 5]):
                frames = sequences[video_id]
                for column, position in enumerate(spaced_indices(len(frames), 3)):
                    source = frames[position]
                    with Image.open(source) as image:
                        image = image.convert("RGB")
                        image.thumbnail((512, 288))
                        sheet.paste(image, (column * 512, row * 320 + 32))
                    number = batch * 5 + row + 1
                    draw.text((column * 512 + 4, row * 320 + 6),
                              f"{number:02d} {video_id} / {source.stem}", fill="black")
                    writer.writerow([number, video_id, source.relative_to(ROOT),
                                     hashlib.sha256(source.read_bytes()).hexdigest()])
            sheet.save(output / f"screen_{batch + 1:02d}.jpg", quality=95)
    print(f"Inventoried {len(sequences)} sequences; previewed 60 candidates in {output}")


def sample_dataset(sequences):
    # Sampling is a frozen milestone. Do not silently reset annotation statuses.
    if (ROOT / "data/manifest.csv").exists():
        raise ValueError("Manifest already exists; preserve the frozen sample instead of resampling")
    review = json.loads((ROOT / "data/screening_review.json").read_text())
    candidates = sorted(sequences)
    random.Random(42).shuffle(candidates)
    splits = {}
    for split in ("train", "validation", "test"):
        for number in review[f"{split}_candidates"]:
            splits[candidates[number - 1]] = split
    prefixes = [video_id.split("-")[0] for video_id in splits]
    if len(set(prefixes)) != 25 or len(splits) != 25:
        raise ValueError("Selected sequences share a source prefix or are repeated")
    pilot_counts = dict(zip(review["pilot_candidates"], [13, 13, 13, 12, 12, 12]))
    records = []
    seen_hashes = set()
    for number, video_id in enumerate(candidates[:60], start=1):
        if video_id not in splits:
            continue
        split = splits[video_id]
        frames = sequences[video_id]
        pilot_positions = set(spaced_indices(20, pilot_counts[number])) if number in pilot_counts else set()
        for sample_index, position in enumerate(spaced_indices(len(frames), 20)):
            source = frames[position]
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest in seen_hashes:
                raise ValueError(f"Duplicate sampled image: {source}")
            seen_hashes.add(digest)
            image_id = f"{number:02d}_{sample_index + 1:02d}"
            target = ROOT / "curated_dataset/images" / split / f"{image_id}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            with Image.open(source) as image:
                image.verify()
            with Image.open(source) as image:
                width, height = image.size
            records.append({"image_id": image_id, "video_id": video_id, "split": split,
                            "source_path": str(source.relative_to(ROOT)),
                            "image_path": str(target.relative_to(ROOT)),
                            "frame_position": position, "sha256": digest,
                            "width": width, "height": height,
                            "pilot": int(sample_index in pilot_positions),
                            "annotation_status": "pending", "qa_status": "pending"})
    with (ROOT / "data/manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (ROOT / "data/candidate_sources.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "video_id", "lighting", "weather_observation",
                         "road_type", "traffic_density", "notes", "assigned_split"])
        for number, video_id in enumerate(candidates[:60], start=1):
            writer.writerow([number, video_id, *review["observations"][str(number)],
                             splits.get(video_id, "not_selected")])
    counts = {split: sum(row["split"] == split for row in records) for split in splits.values()}
    assert counts == {"train": 300, "validation": 100, "test": 100}, counts
    assert sum(row["pilot"] for row in records) == 75
    print(f"Sampled {len(records)} unique images: {counts}; pilot=75")


def restore_samples():
    with (ROOT / "data/manifest.csv").open() as handle:
        records = list(csv.DictReader(handle))
    for row in records:
        source = (ROOT / row["source_path"]).resolve()
        target = (ROOT / row["image_path"]).resolve()
        if not source.is_relative_to(RAW) or not target.is_relative_to(ROOT / "curated_dataset/images"):
            raise ValueError("Manifest path is outside the expected source or curated image folder")
        expected = row["sha256"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Raw source changed: {source}")
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Refusing to overwrite changed curated image: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"Restored/verified {len(records)} images without changing frozen metadata")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previews", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--restore", action="store_true")
    arguments = parser.parse_args()
    if arguments.restore:
        restore_samples()
    else:
        sources = inventory()
        if arguments.previews:
            previews(sources)
        elif arguments.sample:
            sample_dataset(sources)
        else:
            print(f"Inventoried {len(sources)} sequences")