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
human-drawn or independently audited labels. No annotation has been produced yet.
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
- All annotations remain pending. Pending images must never train as negatives.

## Runtime and Completion Policy

- No fixed CPU-time budget was requested. Benchmark before lengthy training.
- Use a local `.venv`; do not install into other projects' environments.
- Record incomplete work rather than fabricate labels, metrics, or completion.
- Commit each validated milestone. Track all reports, including figures and
  prediction images. Raw datasets, curated image collections, checkpoints,
  environments, and run logs remain ignored.