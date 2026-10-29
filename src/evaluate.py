"""Freeze a validation-selected model, then evaluate the held-out test once."""

import argparse
import hashlib
import json

import torch

from src.data import VehicleDataset
from src.metrics import evaluate_predictions
from src.model import VehicleDetector
from src.prepare_data import ROOT
from src.train import predict_dataset, save_overlays, seed_everything


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    path = ROOT / "reports/frozen_config.json"
    if path.exists() or (ROOT / "reports/final_test_result.json").exists():
        raise ValueError("An evaluation configuration is already frozen; do not overwrite it")
    results = [json.loads((ROOT / f"reports/{name}_seed42.json").read_text())
               for name in ("baseline", "augmentation")]
    if any(result.get("split") != "validation" for result in results):
        raise ValueError("Selection must use validation results only")
    selected = max(results, key=lambda result: result["ap50"])
    checkpoint = ROOT / "checkpoints" / selected["config"]["name"] / "best.pt"
    frozen = {"checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha256(checkpoint),
              "validation_ap50": selected["ap50"],
              "operating_threshold": selected["operating_threshold"],
              "config": selected["config"], "manifest_sha256": sha256(ROOT / "data/manifest.csv"),
              "annotation_sha256": {split: sha256(ROOT / f"curated_dataset/annotations/{split}.json")
                                    for split in ("train", "validation", "test")}}
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"Frozen {selected['config']['name']} using validation AP@0.5 only")


def evaluate_test():
    output = ROOT / "reports/final_test_result.json"
    if output.exists():
        raise ValueError("Test results already exist; do not repeatedly evaluate and tune")
    frozen = json.loads((ROOT / "reports/frozen_config.json").read_text())
    checkpoint_path = (ROOT / frozen["checkpoint"]).resolve()
    if not checkpoint_path.is_relative_to(ROOT):
        raise ValueError("Checkpoint is outside project")
    if sha256(checkpoint_path) != frozen["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after freeze")
    if sha256(ROOT / "data/manifest.csv") != frozen["manifest_sha256"]:
        raise ValueError("Dataset manifest changed after freeze")
    for split, expected in frozen["annotation_sha256"].items():
        if sha256(ROOT / f"curated_dataset/annotations/{split}.json") != expected:
            raise ValueError("Annotations changed after freeze")
    seed_everything(frozen["config"]["seed"], frozen["config"]["threads"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = VehicleDetector().cpu()
    model.load_state_dict(checkpoint["model"])
    dataset = VehicleDataset("test")
    predictions, targets = predict_dataset(model, dataset, frozen["config"]["batch_size"])
    result = evaluate_predictions(predictions, targets, frozen["operating_threshold"])
    result.update({"split": "test", "frozen": frozen,
                   "limitation": "AI-produced labels lack independent human audit",
                   "ap_definition": "all-points interpolated AP@0.5; floor 0.01; NMS IoU 0.45; max 100/image"})
    output.write_text(json.dumps(result, indent=2) + "\n")
    save_overlays(dataset, predictions, frozen["operating_threshold"], ROOT / "reports/predictions/final_test")
    print(json.dumps({key: value for key, value in result.items() if key not in ("pr_curve", "frozen")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "test"])
    arguments = parser.parse_args()
    if arguments.action == "freeze":
        freeze()
    else:
        evaluate_test()