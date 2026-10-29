# Vehicle Detection From Scratch

A compact computer-vision project for learning the complete object-detection workflow: data selection, bounding-box annotation, leakage-safe splitting, model training, and honest evaluation.

The goal is not to reproduce a production detector or beat YOLO. The goal is to understand what an object detector learns and how to evaluate it fairly.

## Project Question

> How well can a custom PyTorch detector locate vehicles in driving-video sequences it has never seen, when trained on manually annotated frames and evaluated with video-level train/validation/test splits?

## Scope

This is a supervised object-detection project built with PyTorch. The detector will be a small SSD-style model built from custom components.

Included:

- Curating a small dataset from raw driving-video frames.
- Writing annotation rules before broad labeling.
- Drawing bounding boxes manually in CVAT.
- Creating train, validation, and test splits by source video.
- Building a small CNN-based detector, anchor matching, losses, decoding, and non-maximum suppression.
- Measuring AP, precision, recall, and common failure cases.

Not included:

- Pretrained backbones, YOLO, or ready-made detection models.
- The full BDD100K dataset.
- Segmentation. It is a separate project with pixel masks, different losses, and different metrics.
- Semi-supervised pseudo-labeling. That can be a later extension after the supervised baseline is reliable.

## Dataset

The source dataset is the BDD100K tracking/MOT subset, which contains driving-video sequences. The project dataset is a manually selected and labeled subset of those raw sequences.

Do not inspect or use BDD100K's supplied detection annotations while choosing frames or producing this project's labels. They may be used later for a clearly documented annotation-quality audit, but never silently used to replace manual labels.

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

Start with a pilot of 75 images from six videos. The pilot exists to test the annotation rules and estimate labeling time. Review and revise rules once after the pilot, then freeze the rules before labeling the full 500-image dataset.

## Class Plan

Build the project in stages. Each stage uses the same frozen video split.

| Stage | Classes                     | Purpose                                                                  |
| ----- | --------------------------- | ------------------------------------------------------------------------ |
| 1     | `vehicle`                 | Learn box localization without class imbalance complicating the problem. |
| 2     | `car`, `large_vehicle`  | Learn classification while combining rare buses and trucks.              |
| 3     | `car`, `truck`, `bus` | Separate vehicle types if each class has sufficient examples.            |

For Stage 1, every annotated car, truck, and bus becomes `vehicle`. For Stage 2, truck and bus become `large_vehicle`.

Do not start Stage 3 unless the validation and test splits each contain enough truck and bus examples to make class-level AP meaningful. If they do not, report the limitation and keep the two-class version. Never fix this by distributing near-duplicate frames from one video across several splits.

## Annotation Protocol

Use CVAT to draw axis-aligned boxes. Export COCO JSON as the master annotation format. Learn the YOLO text format through a conversion script, but do not use YOLO to train this detector.

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

Validation data decides architecture details, learning rate, augmentation, confidence threshold, NMS threshold, and stopping epoch. The test videos remain untouched until all decisions are frozen.

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

## Final Claim

> I curated a driving-video dataset of approximately 500 images from 25 source sequences, defined and applied bounding-box annotation rules, audited labels in a second pass, used leakage-safe video-level splits, trained a custom SSD-style detector, and analyzed its behavior on unseen driving videos.

## Future Extensions

After the supervised detector is stable:

- Expand the dataset toward 750, then 2,000-5,000 images by adding diverse source videos rather than near-duplicate frames.
- Compare binary, two-class, and three-class detection fairly.
- Add a separate segmentation project for drivable area or road masks.
- Run a semi-supervised pseudo-labeling experiment with a separate pool of unlabeled videos and a fixed untouched test set.


## need to extend at end to this : 
Semi_supervised_vehicle_detection is misleading because Version 1 uses only labeled data:

image + manually drawn bounding boxes
    -> train detector

That is supervised learning.

Semi-supervised detection would require a separate unlabeled pool and a process like:

small labeled dataset
    -> train initial detector
    -> predict boxes on unlabeled images
    -> keep high-confidence pseudo-labels
    -> retrain
    -> evaluate on untouched labeled test videos

Your current project may eventually become semi-supervised, but its first and most important version is a supervised detector. A name such as vehicle_detection_from_scratch leaves room for later expansion without describing the wrong learning setup.