# Version 1 Status and Resume Guide

**Last updated:** 2026-10-30  
**State:** In progress; no validation or test performance has been produced.

This file is the detailed pause/resume record for Version 1. Update it whenever
work pauses, and commit it alongside the relevant code, annotations, manifests,
and reports.

## Goal and Constraints

- Complete Version 1 as a **binary `vehicle` detector**. Cars, buses, and trucks
  are all one foreground class. Multi-class stages are out of scope here.
- Use only raw frames from `dataset/images20-track-train-1` for selection and
  labels. Do not inspect or use supplied BDD100K annotations.
- Do not use pretrained detectors. Train the small custom detector from random
  initialization.
- Run CPU-only using the project-local `.venv`; do not use GPU or install into
  another project environment.
- Keep project work inside `vehicle_detection_from_scratch/`. Preserve raw data.
- Track reports, figures, prediction overlays, code, annotations, and manifests
  in local Git. Raw images, checkpoints, environments, and run logs are ignored.
- Labels are AI-produced and visually self-reviewed, not independently audited
  by a human. Do not claim human annotation or independent ground truth.

## Frozen Dataset

Selection and split assignment are committed. The source pool contains 200 raw
sequence directories. Sixty candidates were visually screened from three spaced
raw frames each; one video per shared ID prefix was selected as a conservative
sequence-isolation rule. This is not proof that the underlying journeys are
unrelated.

| Split | Videos | Selected images | Reviewed | Draft | Pending |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 15 | 300 | 51 | 12 | 237 |
| Validation | 5 | 100 | 0 | 0 | 100 |
| Test | 5 | 100 | 0 | 0 | 100 |
| **Total** | **25** | **500** | **51** | **12** | **437** |

The manifest stores source video, source frame position, split, pilot flag,
dimensions, paths, and SHA-256. All selected images decoded, and no duplicate
sample hash was found. The 75-image pilot is included in the 300 training images:

- 51 pilot images have AI-reviewed labels, totaling **375 vehicle boxes**, across
  four training videos.
- 12 dusk images from a fifth pilot video have **draft-only labels (109 boxes)**.
  Overlay review exposed speculative boxes over houses and empty roadway. Those
  were removed; the remaining labels are not training-ready.
- The last 12 pilot images from the sixth training video are not yet annotated.
- The other 425 selected images are not yet annotated.
- Total not eligible as reviewed labels: 449 images (12 draft + 437 pending).
- All 51 reviewed images are in training. Validation and test currently have no
  reviewed annotations.

The full training/validation loader refuses any split with an unreviewed image.
Draft labels must never be treated as background examples. The dedicated
`pilot_only=True` loader is for training-only diagnostics and cannot load
validation or test data.

## Detector and Verification

Implemented in simple Python/PyTorch modules:

- Two-feature-map custom CNN detector, 4,500 anchors, binary foreground/background.
- Anchor generation, IoU, forced-positive matching, box encoding/decoding,
  hard-negative mining, Smooth L1 localization loss, and CPU NMS.
- Aspect-preserving letterbox preprocessing and training-only flip/color changes.
- Single-class all-points AP@0.5, threshold selection on validation, prediction
  overlays, validation-only checkpoint freeze, and a guarded one-time test path.
- COCO export records source provenance and whether labels received an
  independent human audit (they did not).
- Dataset restoration checks manifest hashes and does not resample or overwrite
  changed curated images.

**Latest test result:** 21 focused tests pass with the project interpreter:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover split isolation/hashes, coordinate and size validation, refusal
to train on pending labels, image transforms, geometry, loss/backpropagation,
NMS, AP matching, deterministic initialization, and the controlled ablation.

The installed PyTorch is `2.5.1+cpu`; `torch.version.cuda` is `None`. No GPU was
used.

## Training-Only Diagnostic

`reports/pilot_overfit.json` records a 300-step CPU diagnostic on four reviewed
training images (`01_01`, `01_03`, `01_04`, `01_06`), seed 42, Adam at 0.001, and
two CPU threads:

- Loss: 8.3101 at step 1 to 0.0345 at step 300.
- Elapsed time: approximately 21 seconds on this machine.
- At score threshold 0.5: 35 training targets matched, 2 false positives, and no
  missed training references.
- The visual prediction review found an extra large box and a small spurious
  prediction on `01_01`.

This demonstrates that optimization can memorize four related training frames.
It is **not** a baseline, validation result, test result, or evidence of
generalization to unseen videos. Do not use its metrics to select model settings.

The synthetic CPU benchmark measured approximately 0.13 seconds per eight-image
training step, about five seconds for one 300-image training epoch before image
loading and validation. Full-run duration has not been measured. Training is
expected to be short compared with annotation.

## Current Commits

The repository is local-only; no remote is configured or used. Recent milestones:

- `b23f945` Review fifth pilot video and add detailed pause-resume status.
- `c39ef5e` Save fourth pilot video as draft pending dusk-label review.
- `321f54f` Review third pilot video with 73 additional vehicle boxes.
- `4ea03d9` Review second pilot video with 123 additional vehicle boxes.
- `cd26525` Verify real-image CPU overfit diagnostic and reproducible data restoration.
- `0cf002f` Review first 13 training pilot images with 126 vehicle boxes.
- `1abff7a` Add reproducible CPU training and guarded evaluation pipeline.
- `9aaf6b6` Track reports including screening figures and annotation overlays.
- `34dd36b` Implement and test compact CPU-only SSD-style detector core.
- `90fc7cd` Freeze screened video splits and reproducible 500-frame sample.
- `3190725` Initialize CPU-only binary detection project and document workflow.

## Exact Next Steps

1. Revisit the 12 draft dusk frames from candidate 08. Inspect the raw image and
   zoom/overlay; correct or remove each unsupported box. If the scene remains too
   ambiguous, exclude and replace that frame with another frame from the same
   training video according to the frozen sampling/replacement rules. Do not
   promote the entire batch simply because the JSON validates.
2. Annotate and review the last 12 pilot frames from candidate 40. Save their
   first-pass boxes, review overlays, and corrections. Keep pilot membership at
   75 unless a documented image exclusion requires a same-split replacement.
3. Review all 75 pilot images together. Revise annotation rules once, freeze the
   resulting version, and correct any pilot labels affected by that revision.
4. Only after the pilot rules are frozen, annotate and second-pass review the
   remaining 225 training images, 100 validation images, and 100 test images.
   Preserve video-level splits; never move test images based on predictions.
5. Run dataset checks: all images decode, boxes are in bounds and at least 20x20
   source pixels, all in-scope vehicles are labeled, hashes match, and split/video
   isolation remains intact. Require all labels reviewed before full-data export.
6. Confirm the model can overfit a small diverse training subset. Keep this
   diagnostic separate from validation and test metrics.
7. Train the baseline and augmentation experiment on CPU with the same frozen
   data and seed. Inspect and save validation prediction overlays after each
   experiment. Record false positives, missed vehicles, duplicate detections,
   small/distant targets, and glare/occlusion errors.
8. Select the model and confidence threshold using validation only. Freeze the
   checkpoint/configuration and hashes before the final test evaluation.
9. Run the guarded test evaluation once. Save AP@0.5, precision/recall, counts,
   curves, representative errors, and the AI-label limitation.
10. Update this file, `reports/progress.md`, README, and workflow notes; run all
    tests; commit each completed milestone locally.

## Commands

From this directory, restore ignored curated image copies from the committed
manifest without changing the sample or labels:

```bash
.venv/bin/python -m src.prepare_data --restore
.venv/bin/python -m src.annotations export
.venv/bin/python -m unittest discover -s tests -v
```

The full-data completion gate currently fails by design:

```bash
.venv/bin/python -m src.annotations export --require-complete
```

Do not run the baseline until all training and validation labels are reviewed.
Do not run final test evaluation until the validation-selected model is frozen.