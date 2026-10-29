"""CPU training and validation. Refuses incomplete labels and test-result reuse."""

import argparse
import json
import random
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from src.data import VehicleDataset, collate_batch
from src.geometry import detection_loss, detections
from src.metrics import choose_threshold, evaluate_predictions
from src.model import VehicleDetector
from src.prepare_data import ROOT


def seed_everything(seed=42, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def predict_dataset(model, dataset, batch_size=8):
    model.eval()
    predictions, targets = {}, {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_batch)
    for images, boxes, identifiers in loader:
        logits, offsets = model(images.cpu())
        for index, image_id in enumerate(identifiers):
            predictions[image_id] = detections(logits[index], offsets[index], model.anchors)
            targets[image_id] = boxes[index]
    return predictions, targets


def save_overlays(dataset, predictions, threshold, output):
    output.mkdir(parents=True, exist_ok=True)
    # Fixed spaced examples prevent choosing only attractive predictions.
    indices = sorted(set(np.linspace(0, len(dataset) - 1, min(6, len(dataset))).round().astype(int)))
    for index in indices:
        tensor, reference, image_id = dataset[index]
        array = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(array).resize((640, 640))
        draw = ImageDraw.Draw(image)
        for box in reference:
            draw.rectangle((box * 640).tolist(), outline="lime", width=2)
        boxes, scores = predictions[image_id]
        for box, score in zip(boxes, scores):
            if float(score) >= threshold:
                coordinates = (box * 640).tolist()
                draw.rectangle(coordinates, outline="red", width=2)
                draw.text((coordinates[0], coordinates[1]), f"{float(score):.2f}", fill="red")
        image.save(output / f"{image_id}.png")


def benchmark(steps=20):
    if steps < 2:
        raise ValueError("Benchmark requires at least two measured steps")
    seed_everything()
    model = VehicleDetector().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    images = torch.zeros(8, 3, 320, 320)
    images[:, :, 120:180, 100:180] = 1.
    targets = [torch.tensor([[100 / 320, 120 / 320, 180 / 320, 180 / 320]]) for _ in range(8)]
    losses = []
    durations = []
    for _ in range(steps + 2):
        start = time.perf_counter()
        logits, offsets = model(images)
        loss = detection_loss(logits, offsets, model.anchors, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        durations.append(time.perf_counter() - start)
    seconds_per_batch = sum(durations[2:]) / steps
    result = {"kind": "synthetic CPU optimization smoke test, NOT dataset performance",
              "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
              "threads": torch.get_num_threads(), "batch_size": 8, "measured_steps": steps,
              "seconds_per_training_batch": seconds_per_batch,
              "estimated_300_image_training_seconds_per_epoch": seconds_per_batch * 38,
              "estimate_excludes": "image loading, augmentation, validation, checkpointing and annotation",
              "first_loss": losses[0], "last_loss": losses[-1]}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/cpu_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def overfit_pilot(steps):
    if steps < 2:
        raise ValueError("Overfit check needs at least two optimization steps")
    output = ROOT / "checkpoints/pilot_overfit"
    if output.exists():
        raise ValueError("Pilot overfit run already exists; preserve the recorded result")
    seed_everything()
    dataset = VehicleDataset("train", pilot_only=True)
    subset = Subset(dataset, list(range(min(4, len(dataset)))))
    images, targets, identifiers = collate_batch([subset[index] for index in range(len(subset))])
    model = VehicleDetector().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        logits, offsets = model(images)
        loss = detection_loss(logits, offsets, model.anchors, targets)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite pilot loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            record = {"step": step, "loss": float(loss.detach())}
            history.append(record)
            print(json.dumps(record), flush=True)
    predictions, references = predict_dataset(model, subset)
    metrics = evaluate_predictions(predictions, references, operating_threshold=0.5)
    output.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "steps": steps, "seed": 42,
                "training_image_ids": identifiers}, output / "last.pt")
    result = {"kind": "four-image training-only overfit diagnostic, NOT validation or test performance",
              "device": "cpu", "seed": 42, "threads": 2, "steps": steps,
              "training_image_ids": identifiers, "history": history,
              "elapsed_seconds": time.perf_counter() - started,
              "training_subset_metrics": metrics,
              "annotation_limitation": "AI-produced labels, no independent human audit"}
    (ROOT / "reports/pilot_overfit.json").write_text(json.dumps(result, indent=2) + "\n")
    save_overlays(subset, predictions, 0.5, ROOT / "reports/predictions/pilot_overfit")
    print(f"Training-only overfit check finished; final loss={history[-1]['loss']:.4f}")


def train(config_path):
    if (ROOT / "reports/final_test_result.json").exists():
        raise ValueError("Final test already evaluated. Do not tune against this test set.")
    config = json.loads(config_path.read_text())
    seed_everything(config["seed"], config["threads"])
    training = VehicleDataset("train", augment=config["augment"])
    validation = VehicleDataset("validation")
    run_name = config["name"]
    if not run_name.replace("_", "").isalnum():
        raise ValueError("Run name must contain only letters, numbers, and underscores")
    output = ROOT / "checkpoints" / run_name
    if output.exists():
        raise ValueError("Run already exists; choose a fresh name instead of overwriting results")
    output.mkdir(parents=True)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True,
                        generator=generator, num_workers=0, collate_fn=collate_batch)
    model = VehicleDetector().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    best_ap, best_epoch, history = -1., 0, []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.
        for images, targets, _ in loader:
            logits, offsets = model(images.cpu())
            loss = detection_loss(logits, offsets, model.anchors, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; stop rather than save invalid weights")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        predictions, references = predict_dataset(model, validation, config["batch_size"])
        result = evaluate_predictions(predictions, references)
        if result["ap50"] is None:
            raise ValueError("Validation contains no vehicle targets")
        record = {"epoch": epoch, "training_loss": total_loss / len(loader),
                  "validation_ap50": result["ap50"], "elapsed_seconds": time.perf_counter() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        if result["ap50"] > best_ap:
            best_ap, best_epoch = result["ap50"], epoch
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, output / "best.pt")
        if epoch - best_epoch >= config["patience"]:
            break
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    predictions, references = predict_dataset(model, validation, config["batch_size"])
    threshold = choose_threshold(predictions, references)
    result = evaluate_predictions(predictions, references, threshold)
    result.update({"config": config, "best_epoch": best_epoch, "history": history,
                   "split": "validation", "annotation_source": "AI visual labels; no independent human audit",
                   "ap_definition": "all-points interpolated AP@0.5; score floor 0.01; NMS IoU 0.45; max 100/image"})
    (ROOT / f"reports/{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
    save_overlays(validation, predictions, threshold, ROOT / "reports/predictions" / run_name)
    print(f"Saved validation metrics and six fixed prediction overlays for {run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--pilot-overfit", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.benchmark:
        benchmark(arguments.steps)
    elif arguments.pilot_overfit:
        overfit_pilot(arguments.steps)
    else:
        path = (ROOT / arguments.config).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Configuration must stay inside the project")
        train(path)