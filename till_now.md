# Work Completed So Far

**Last updated:** 2026-10-30  
**Project state:** In progress; no full-data validation or test result exists.

This document records what has actually been completed for Version 1. For the
live counts, pause/resume checklist, commands, and exact future steps, see
[`status.md`](status.md).

## Agreed Version 1 Scope

- Build one binary object detector with foreground class `vehicle`.
- Use raw images from `dataset/images20-track-train-1` only.
- Do not inspect or use the supplied BDD100K annotations.
- Do not use pretrained detectors; initialize and train the detector ourselves.
- Run training on CPU only, in the project-local Python environment.
- Keep code straightforward, with tests, local commits, and reports tracked in Git.
- Labels are AI-produced and visually self-reviewed. They are not human-drawn or
  independently audited.

## Why These Choices

| Choice | Reason |
| --- | --- |
| Start with one `vehicle` class | This first detector focuses on object localization and avoids adding car/truck/bus class imbalance before the binary pipeline works. |
| Split by source video before sampling | Adjacent frames from one drive are highly correlated. Keeping a whole source video in one split reduces train/validation/test leakage. |
| Screen 60 videos to select 25 | Screening a larger pool gives a chance to choose varied roads, traffic, weather, and lighting instead of accepting 25 arbitrary sequences. |
| Sample spaced frames | Spaced samples reduce near-duplicate frames while retaining conditions across each sequence. Exact seconds between frames were not claimed because timing metadata was not verified. |
| Keep the target at 500 images | 500 is the project's planned, manageable starting dataset, not a technical minimum. A smaller dataset would require revising the stated experiment plan and could make results less informative. |
| Annotate one box per vehicle | A detector needs object locations, not just an image-level “vehicle present” label. Without boxes it cannot learn localization or support AP/IoU evaluation. |
| Keep separate train/validation/test labels | Training boxes teach the model; validation boxes support experiment choices; test boxes provide a final held-out measurement. Test frames can be prepared, but must not guide model decisions. |
| Do not inspect supplied annotations | This preserves the user's requested raw-frame visual-labeling approach and avoids silently substituting existing ground truth. |
| Mark ambiguous labels as draft | Incorrect or missing boxes can teach the detector that a real vehicle is background. Draft/pending states prevent uncertain labels from contaminating training or reported metrics. |
| Review overlays against raw pixels | Coordinate checks catch out-of-bounds or undersized boxes, but only visual inspection can reveal a box on a house, empty road, wrong vehicle, or merged objects. |
| Use CPU-only dependencies and runs | CPU-only execution is an explicit user requirement; a local environment also avoids changing other projects' packages. |
| Keep large data/checkpoints out of Git but track reports | Raw images and model weights are bulky local artifacts. Manifests, annotations, code, figures, and reports are needed to explain and reproduce the work. |

## Project and Repository Setup

- Initialized a Git repository inside `vehicle_detection_from_scratch/` only.
- Added `.gitignore` rules for raw/curated images, checkpoints, run logs, local
  environments, and generated caches.
- Reports, figures, prediction overlays, manifests, annotations, source code,
  tests, and documentation are tracked, as requested.
- Added `flow.md`, `decisions.md`, this file, and the detailed living checkpoint
  `status.md`.
- Created an isolated `.venv` inside this project and installed Pillow, NumPy,
  and the CPU-only PyTorch 2.5.1 build. No GPU was used.
- Original raw data remains in place; no source images were modified.

## Frame Selection and Dataset Preparation

- Inventoried **200** raw video-sequence directories.
- Reproducibly sampled and visually screened **60 candidate videos**, inspecting
  three spaced raw frames per candidate. No supplied annotations were opened.
- Recorded candidate observations, selected video IDs, frame paths, dimensions,
  split assignments, image hashes, and pilot membership.
- Selected **25 videos** and assigned whole videos to splits before sampling:

| Split | Videos | Selected frames |
| --- | ---: | ---: |
| Training | 15 | 300 |
| Validation | 5 | 100 |
| Test | 5 | 100 |
| **Total** | **25** | **500** |

- Sampled 20 positions across each selected sequence. The frame positions are
  reproducible; exact timing in seconds was not verified.
- Applied a conservative rule of one selected sequence per shared ID prefix.
  This reduces some leakage risk but does not prove that the journeys are
  unrelated.
- Verified that all 500 selected frames decode and have unique SHA-256 hashes.
- Implemented manifest-based image restoration. It verifies source hashes and
  restores ignored image copies without resampling or resetting labels.

## Annotation Work

All labels are for one class, `vehicle`. Each image has individual original-image
pixel boxes. Draft and pending images are never treated as reviewed negatives.

| Label state | Images | Boxes | Meaning |
| --- | ---: | ---: | --- |
| AI-reviewed | 75 | 514 | Visually checked against raw frames and overlays; training split only |
| Draft | 0 | 0 | None |
| Pending | 425 | 0 | Remaining train, validation, and test images |
| **Total** | **500** |  |  |

- All 75 reviewed images are spread over six training videos and are part of the
  300-image training split, not extra images.
- Dusk/night overlay reviews removed or corrected boxes on pedestrians, houses,
  empty roadway, headlight glare, and reflections.
- Annotation guideline v0.3 is frozen after the one planned pilot revision.
- The other 225 training frames and all 100 validation and 100 test frames remain
  unannotated.
- Annotation tooling produces coordinate guides, raw-frame views, box overlays,
  validation checks, and COCO JSON exports. Boxes must be at least 20 x 20 source
  pixels and stay within the image.
- Exports record that annotations are AI-produced and lack an independent human
  audit. Low-light glare, occlusion, and tiny distant vehicles are known risks.

**Why full training is still blocked:** 425 images are not in the reviewed state.
Treating them as empty-label images would incorrectly
teach the detector that visible vehicles are background. Validation and test also
need reviewed boxes before their metrics mean anything. This is why the current
pipeline rejects full-data training rather than producing a misleading run.

## Detector Implementation

Implemented a compact detector from random initialization:

- Custom CNN with two feature maps and 4,500 anchors.
- Binary classification output: background and `vehicle`.
- Anchor generation, IoU, matching, forced positive matches, box encoding and
  decoding, hard-negative mining, Smooth L1 localization loss, and NMS.
- Aspect-preserving image resize, CPU data loading, and training-only horizontal
  flip/color augmentation.
- Validation AP@0.5/precision/recall, prediction overlays, threshold selection,
  validation-only checkpoint freeze, and guarded one-time test evaluation.
- Full-data training refuses to start while any image in the requested split has
  pending or draft annotations. The partial pilot loader is training-only.

## Verification and Runs

- The project test suite has **21 passing tests** covering split isolation,
  hashes, box checks, annotation gates, transforms, geometry, matching, NMS,
  loss/backpropagation, AP behavior, deterministic initialization, and ablation
  configuration.
- A synthetic CPU benchmark measured about **0.13 seconds per eight-image
  optimization batch** on this machine. Its approximate five-second estimate for
  one 300-image training epoch excludes loading, augmentation, and validation; it
  is not a full-run measurement.
- Ran a **300-step overfit diagnostic** on four reviewed training images only:
  loss decreased from **8.3101** to **0.0345** in about 21 seconds. At score
  threshold 0.5, it matched 35 training boxes, produced two false positives, and
  missed zero training reference boxes. The prediction overlays were inspected.
- The overfit diagnostic tests that optimization can memorize those four images.
  It is not a baseline, validation result, test result, or evidence of
  generalization.
- The CPU-only PyTorch build reports no CUDA support. No GPU was used.

**Why the overfit diagnostic was run:** it is a cheap check that gradients and
the training loop can fit real reviewed examples before spending time labeling
and training on the full dataset. Its high training score is expected after
memorization and must not be presented as unseen-video performance.

## Git Milestones

The project has local commits for initialization, dataset preparation, annotation
tools, detector implementation, report tracking, training/evaluation safeguards,
the overfit diagnostic, six pilot-video batches, and the frozen pilot rules. The repository
has no remote push as part of this work. The exact recent commit list is in
[`status.md`](status.md).

## Not Completed Yet

- Annotate and review the remaining 425 images under frozen guideline v0.3.
  **Why:** boxes are needed for training, validation-based experiment choices,
  and a held-out test measurement.
- Complete dataset-wide quality and split checks with all labels reviewed.
  **Why:** a credible detector metric depends on valid boxes and leakage-safe
  splits, not only code that runs.
- Train the full-data baseline and augmentation experiment on CPU. **Why:** the
  baseline establishes performance, while the controlled experiment tests one
  change at a time.
- Inspect validation predictions, select settings using validation only, and
  freeze the chosen checkpoint/configuration. **Why:** using test results to tune
  would bias the final estimate.
- Evaluate once on the held-out test set and write final metrics and error
  analysis. **Why:** test data is for the final estimate after choices are fixed;
  error analysis explains where the detector succeeds or fails.

**Therefore Version 1 is not complete and no full detector performance claim is
currently supported.**