# Vehicle Detection From Scratch

## Active Version 1

Version 1 is a CPU-only binary `vehicle` detector. Dataset selection is complete:
500 images from 25 distinct source-prefix groups, split into 300 training,
100 validation, and 100 test images. 51 training-pilot images (375 boxes) are
AI-reviewed; 12 images remain draft and 437 remain pending. Full training is
not complete. See [status.md](status.md) for the current detailed status and
resume checklist.
See [flow.md](flow.md) for the workflow and [decisions.md](decisions.md) for the
execution decisions. Two- and three-class detection below are future extensions,
not deliverables of the active run.

Annotations will be AI-produced from visual inspection of raw frames and reviewed
in a second visual pass, not human-drawn in CVAT. Supplied BDD100K annotations and
pretrained detectors must not be used. Test annotations are prepared using frozen
rules, then held out from model development. No completed-model claim is made.

The CPU implementation passes 21 focused tests. A four-image training-only
overfit check reduced loss from 8.31 to 0.035 over 300 steps. This only tests
optimization and memorization; no validation or test result is available.
See [reports/progress.md](reports/progress.md) for the current completion boundary.

A compact computer-vision project for learning the complete object-detection workflow: data selection, bounding-box annotation, leakage-safe splitting, model training, and honest evaluation.

The goal is not to reproduce a production detector or beat YOLO. The goal is to understand what an object detector learns and how to evaluate it fairly.

## Project Question

> How well can a custom PyTorch detector locate vehicles in unseen driving-video sequences when trained on AI-produced, visually self-reviewed annotations and evaluated with video-level train/validation/test splits?

## Scope

This is a supervised object-detection project built with PyTorch. The detector will be a small SSD-style model built from custom components.

Included:

- Curating a small dataset from raw driving-video frames.
- Writing annotation rules before broad labeling.
- Creating bounding boxes through AI visual inspection, with recorded provenance and a second pass.
- Creating train, validation, and test splits by source video.
- Building a small CNN-based detector, anchor matching, losses, decoding, and non-maximum suppression.
- Measuring AP, precision, recall, and common failure cases.

Not included:

- Pretrained backbones, YOLO, or ready-made detection models.
- The full BDD100K dataset.
- Segmentation. It is a separate project with pixel masks, different losses, and different metrics.
- Semi-supervised pseudo-labeling. That can be a later extension after the supervised baseline is reliable.

## Dataset

The source dataset is the BDD100K tracking/MOT subset. The project subset is selected by AI visual screening of raw sequences, with AI-produced labels planned for the selected frames.

Do not inspect or use BDD100K's supplied annotations for selection, labeling, or this run's evaluation. Do not silently substitute supplied labels for the project's own annotations.

### Version 1 Data Protocol

| Item                       |                   Target |
| -------------------------- | -----------------------: |
| Source videos              |                       25 |
| Frames per video           | 20, sampled at intervals |
| Total images               |                about 500 |
| Training videos / images   |           15 / about 300 |
| Validation videos / images |            5 / about 100 |
| Test videos / images       |            5 / about 100 |

The split is created at the video level before frames are sampled or labeled:

```text
25 source videos
  -> 15 training videos
  ->  5 validation videos
  ->  5 test videos
```

Frames from one video must never appear in more than one split. Adjacent video frames are visually similar; splitting them randomly would leak information and make validation or test performance look better than it really is.

Sample frames at regular intervals, such as every one or two seconds, rather than taking consecutive frames. Aim for variation in lighting, weather, traffic density, occlusion, vehicle size, and road type.

### Source Selection and Class Coverage

Do not select 25 random videos and hope that buses and trucks appear often enough. First preview raw footage from a larger candidate pool, such as 50-60 videos. For each candidate, inspect a few well-spaced frames and record a simple source-level screen: `cars visible`, `trucks visible`, `buses visible`, plus conditions such as night or rain. Do not use the supplied BDD100K annotations for this screen.

Use that screen to assign entire videos to train, validation, and test. Every split should contain several distinct source videos with trucks and several with buses; never put all rare-class videos in training. The target for Stage 3 is at least 100 truck boxes and 100 bus boxes from at least five videos each in training, and at least 30 boxes from at least three videos each in validation and test. These are minimum planning gates, not a reason to leak frames across splits.

After the pilot annotations, count boxes by class and split before committing to the full dataset. If the truck or bus targets cannot be reached naturally, retain the `car` versus `large_vehicle` version rather than claiming a reliable three-class detector.

### Pilot Before Version 1

Start with 75 training images from six training videos, included in the 500-image total. Review the pilot and revise rules once, then freeze them before labeling validation or test images.

## Class Plan

Only Stage 1 is in the active Version 1 scope. Stages 2 and 3 are future work and would reuse the frozen video split. Binary annotation requires only `vehicle`, not separate vehicle-type labels. The following class-coverage targets apply to future multiclass work, not the current binary dataset.

| Stage | Classes                     | Purpose                                                                  |
| ----- | --------------------------- | ------------------------------------------------------------------------ |
| 1     | `vehicle`                 | Learn box localization without class imbalance complicating the problem. |
| 2     | `car`, `large_vehicle`  | Learn classification while combining rare buses and trucks.              |
| 3     | `car`, `truck`, `bus` | Separate vehicle types if each class has sufficient examples.            |

For Stage 1, every annotated car, truck, and bus becomes `vehicle`. For Stage 2, truck and bus become `large_vehicle`.

Do not start Stage 3 unless the validation and test splits each contain enough truck and bus examples to make class-level AP meaningful. If they do not, report the limitation and keep the two-class version. Never fix this by distributing near-duplicate frames from one video across several splits.

## Annotation Protocol

Create axis-aligned boxes by visual inspection and store COCO JSON as the master annotation format. Save the producing agent, review status, and image hashes. Optional future human review can use CVAT. Do not use YOLO to produce labels or train this detector.

Each label must follow these rules:

- Annotate all in-scope cars, trucks, and buses in every selected image.
- Draw a tight box around visible vehicle pixels. Do not include shadows or reflections.
- Annotate an occluded or truncated vehicle only when its class is clear.
- Use a minimum box size of 20 by 20 pixels in the original image.
- Treat sedans, SUVs, hatchbacks, vans, and pickups as `car` unless the rules are updated before the dataset freeze.
- Label a truck with an attached trailer as one `truck` box when it is visually one vehicle unit.
- Exclude an image from the dataset when ambiguity would make the label inconsistent. Record the reason in the dataset manifest.
- Do not leave an in-scope vehicle unlabeled. During training, an unlabeled target is incorrectly treated as background.

### Quality Control

After initial labeling:

1. Make a complete second pass for missed boxes, poor boxes, and wrong classes.
2. Re-label a random 10% sample independently when possible.
3. Compare box counts and disagreements, then document corrections.
4. Record removed images and the reason for removal.

## Dataset Metadata

Maintain a manifest such as:

```text
image_id,video_id,timestamp,split,annotation_status,qa_status,removal_reason
000001,video_003,00:00:12.0,train,complete,reviewed,
```

Store the selected video IDs, sampled frame indices, split assignments, annotation version, and random seeds. These records make the experiment reproducible and demonstrate that the test set was not used to guide development.

## Detector Design

The model is a compact anchor-based, SSD-style detector. It is intentionally small enough to study and train on the curated dataset.

```text
320 x 320 image
  -> custom CNN backbone
  -> two feature maps for different object sizes
  -> anchors at each feature-map location
  -> classification head: background + target classes
  -> box-regression head: four offsets per anchor
  -> score filtering and non-maximum suppression
  -> final detections
```

The following components are implemented for learning purposes:

- A custom CNN backbone.
- Anchor generation.
- Intersection over Union (IoU).
- Matching anchors to ground-truth boxes.
- Box encoding and decoding.
- Classification and box-regression heads.
- Classification and localization losses.
- Non-maximum suppression (NMS).
- Detection evaluation.

PyTorch autograd, optimizers, tensors, `DataLoader`, and standard geometry utilities may be used. Pretrained models and ready-made object detectors are out of scope.

### Training Targets and Losses

Each anchor is compared with the ground-truth boxes using IoU:

$$
\operatorname{IoU}(A, B) =
\frac{\operatorname{area}(A \cap B)}
{\operatorname{area}(A \cup B)}
$$

Initial matching rules:

- Positive anchor: IoU at least 0.5 with a ground-truth box.
- Negative anchor: IoU below 0.4 with every ground-truth box.
- Ignored anchor: IoU from 0.4 up to 0.5.

Use cross-entropy for the class prediction, including a `background` class, and Smooth L1 loss for box offsets on positive anchors only. Because background anchors greatly outnumber positive anchors, implement hard-negative mining or another documented background-sampling strategy.

## Evaluation Protocol

Validation data decides architecture details, learning rate, augmentation, confidence threshold, NMS threshold, and stopping epoch. Test data may be sampled, annotated, and checked under frozen data rules, but must not guide model development. Test evaluation occurs only after model decisions are frozen. Self-reviewed AI labels are not independently verified ground truth.

Report:

- AP at IoU threshold 0.5 for every class.
- mAP@0.5 across classes.
- Precision-recall curves.
- Number of detections, false positives, and missed targets.
- Qualitative examples: correct detections, duplicate detections, small vehicles, occluded vehicles, and class confusions.

A predicted box is correct at IoU 0.5 only when it has the correct class, overlaps an unmatched ground-truth box by at least 0.5, and is not a duplicate prediction for that object.

Run at least one controlled ablation, such as comparing no augmentation with horizontal flip plus modest color jitter. Change one decision at a time and select the final configuration using validation results only.

## Milestones

1. **Pilot dataset**: choose six source videos, sample 75 frames, write and test annotation rules.
2. **Dataset freeze**: select 25 source videos, lock the video-level split, label about 500 images, complete quality control, and save the manifest.
3. **Geometry utilities**: implement and test box conversions, IoU, encoding/decoding, matching, and NMS on small known examples.
4. **Binary detector**: train the `vehicle` detector and inspect validation predictions after each experiment.
5. **Two-class detector**: train `car` versus `large_vehicle` if the data contains enough large vehicles.
6. **Three-class detector**: add truck and bus only when per-split counts support valid evaluation.
7. **Final evaluation**: freeze choices, evaluate once on held-out test videos, and write error analysis.

## Expected Limitations

A 500-image dataset will be small and imbalanced. The detector may miss small, distant, occluded, night-time, or uncommon vehicles. These are not failures of the project; they are the error categories to measure and explain.

A successful project is not defined by beating a benchmark. It is defined by a credible and reproducible answer to the project question.

## Target Final Claim (Not Yet Achieved)

> I curated a driving-video dataset of approximately 500 images from 25 source sequences using AI-produced, visually self-reviewed labels, used video-level splits, trained a custom SSD-style binary detector on CPU, and analyzed its behavior on held-out videos. The labels lack an independent human audit.

## Future Extensions

After the supervised detector is stable:

- Expand the dataset toward 750, then 2,000-5,000 images by adding diverse source videos rather than near-duplicate frames.
- Compare binary, two-class, and three-class detection fairly.
- Add a separate segmentation project for drivable area or road masks.
- Run a semi-supervised pseudo-labeling experiment with a separate pool of unlabeled videos and a fixed untouched test set.


## Reproduce the Current State

Run from this project directory with Python 3.12. Raw tracking frames must be
present at the source paths recorded in `data/manifest.csv`; they are not in Git.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.prepare_data --restore
.venv/bin/python -m src.annotations export
.venv/bin/python -m unittest discover -s tests -v
```

`--restore` verifies original hashes and copies missing curated images without
resampling or resetting labels. It refuses to overwrite a changed curated image.

The CPU timing check is independent of dataset-label completion:

```bash
.venv/bin/python -m src.train --benchmark --steps 20
```

The training-only pilot check has already run locally. On a fresh checkout with
restored images and reviewed pilot labels, reproduce it with:

```bash
.venv/bin/python -m src.train --pilot-overfit --steps 300
```

It refuses to overwrite an existing `checkpoints/pilot_overfit` run. Checkpoint
weights remain local; its report and prediction overlays are tracked in Git.
Expect similar optimization results in the pinned CPU environment, not identical
wall-clock times or guaranteed bitwise results on different hardware.

## Full Training After Annotation Completion

These commands are deliberately blocked while labels remain incomplete:

```bash
.venv/bin/python -m src.annotations export --require-complete
.venv/bin/python -m src.train --config configs/baseline.json
.venv/bin/python -m src.train --config configs/augmentation.json
```

Visually inspect the saved validation predictions after each experiment and
record findings. Only after selecting and freezing choices from validation:

```bash
.venv/bin/python -m src.evaluate freeze
.venv/bin/python -m src.evaluate test
```

Version 1 is supervised learning with AI-produced labels, not semi-supervised
pseudo-labeling. There is no separate unlabeled training pool in this run.