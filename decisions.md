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

The proposed autonomous route uses AI-produced bounding boxes based on visual
inspection of raw frames, followed by a visual second pass. These are not
human-drawn or independently audited labels. No annotation has been produced yet.
Confirmation of this documentation change is pending; the existing README still
describes the original manual-CVAT plan.

## Evaluation Boundary

Test data may be sampled, annotated, and checked using frozen dataset rules.
It must not guide architecture, thresholds, augmentation, or stopping decisions.
Use validation predictions after each experiment; evaluate on test only after
the final configuration is frozen. Self-reviewed AI test labels limit the
strength of reported performance claims.

## Pending Before the Full Run

- Confirm AI-produced annotation provenance in place of manual CVAT annotation.
- Set a CPU training-time budget and accept that it may limit experiment count.
- Define the fallback if labels cannot be completed reliably or the full run
  exceeds the available session or compute budget: record incomplete work rather
  than fabricate labels, metrics, or completion.