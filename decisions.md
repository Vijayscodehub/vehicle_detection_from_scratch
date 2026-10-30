# Version 1 Decisions

## Agreed Execution Scope

- Work only inside this project directory; preserve the raw dataset.
- Use CPU only, not GPU.
- Implement the binary `vehicle` detector, not the later multiclass stages.
- Keep code straightforward with useful explanatory comments.
- Use local Git commits at validated milestones; do not push to a remote.
- Target 500 images: 300 training, 100 validation, and 100 test, from 25 videos.
- Assign whole videos to splits before final frame sampling.
- Include the 75-image training pilot within the 300 training images.
- Do not inspect or reuse supplied BDD100K annotations.

## Annotation Provenance

The authorized autonomous route uses AI-produced bounding boxes based on visual
inspection of raw frames, followed by a visual second pass. These are not
human-drawn or independently audited labels. Annotation work is in progress;
the manifest distinguishes pending, draft, and AI-reviewed records.
The user authorized proceeding without supplied annotations. The active README
now records AI provenance rather than claiming manual CVAT annotation.

## Evaluation Boundary

Test data may be sampled, annotated, and checked using frozen dataset rules.
It must not guide architecture, thresholds, augmentation, or stopping decisions.
Use validation predictions after each experiment; evaluate on test only after
the final configuration is frozen. Self-reviewed AI test labels limit the
strength of reported performance claims.

## Sampling Completed

- Seed 42 selected a reproducible 60-sequence screening pool from 200 sequences.
- Three spaced raw frames per candidate were visually inspected.
- Selected 25 videos for scene diversity, with one video per shared ID prefix.
  This is a conservative grouping precaution, not proof of unrelated journeys.
- Exact image SHA-256 checks found no duplicates among the 500 samples.
- Sample equally spaced frame positions including both endpoints. Timing metadata
  is unverified, so do not claim a specific number of seconds between frames.
- Every selected image decodes; the manifest records paths, dimensions, hashes,
  source-frame positions, splits, and pilot membership.
- Pending images must never train as negatives. Full-data training refuses
  any unreviewed image. A separate, explicitly named training-only diagnostic
  may use the reviewed subset of the training pilot.

## Runtime and Completion Policy

- No fixed CPU-time budget was requested. Benchmark before lengthy training.
- Use a local `.venv`; do not install into other projects' environments.
- Record incomplete work rather than fabricate labels, metrics, or completion.
- Commit each validated milestone. Track all reports, including figures and
  prediction images. Raw datasets, curated image collections, checkpoints,
  environments, and run logs remain ignored.

## First Pilot Batch and CPU Diagnostic

- 39 pilot images from three training videos are AI-reviewed, totaling 322
  boxes. These labels remain subject to the eventual full-pilot rule freeze.
- Source images, coordinate guides, overlay reviews, and selected raw-image
  zooms were inspected. Corrections include missed objects, a vegetation false
  label, undersized fragments, overlapping targets, and box boundaries.
- Apply the current emergency-vehicle exclusion only when emergency identity
  is visually supported. Do not infer identity merely from neighboring frames.
- A separate four-image, 300-step diagnostic uses only reviewed training images
  `01_01`, `01_03`, `01_04`, and `01_06`, seed 42, Adam at 0.001, and two CPU threads.
- Diagnostic loss changed from 8.3101 to 0.0345 in about 21 seconds. At threshold
  0.5, it matched 35 training targets and produced two false positives. These are
  memorization diagnostics, not generalization results or hyperparameter evidence.
- Installed PyTorch is `2.5.1+cpu`; its CUDA build is `None`. No GPU was used.
- The complete baseline, augmentation experiment, validation inspection, final
  model freeze, and held-out evaluation remain pending on completed annotations.