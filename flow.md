# Version 1 Workflow

## Scope

Binary object detection only: cars, trucks, and buses share the label `vehicle`.
Use the local tracking frames, CPU execution, simple Python/PyTorch code, and
documented seeds. Do not use pretrained detectors or supplied BDD100K annotations.
All project changes and outputs stay inside this directory.

## Planned Steps

1. Verify the CPU environment and inventory raw sequences without modifying them.
2. Visually screen candidate videos and record source-level conditions.
3. Freeze 15 training, 5 validation, and 5 test video assignments.
4. Sample 20 spaced frames per video and save a reproducible manifest.
5. Visually annotate 75 training images from six training videos as a pilot.
6. Review the pilot, freeze rules, and annotate/review the remaining images.
7. Validate COCO annotations, coordinates, image integrity, and split isolation.
8. Test simple detector geometry, losses, and a tiny-subset overfit run on CPU.
9. Train a baseline and a controlled augmentation experiment; inspect validation
   prediction overlays after each experiment and record actual results.
10. Freeze the chosen configuration before final held-out test evaluation.
11. Document metrics, failure cases, exact commands, dependencies, and limitations.

## Reproducibility and Commits

Commit each completed, validated milestone locally. Track code, annotations,
manifests, configuration, all reports (including figures and predictions), and
documentation. Keep raw images, checkpoints, environments, and run logs out of
Git. Record source paths and
checksums so ignored data can be identified. CPU training remains subject to a
user-approved time budget; do not promise an accuracy or successful convergence.

## Status

Repository initialization, raw-frame inventory, visual screening, video split,
and 500-frame sampling are complete. All labels remain pending. No detector has
been trained or evaluated. The training pilot contains exactly 75 of the selected
training images, spread over six training videos.