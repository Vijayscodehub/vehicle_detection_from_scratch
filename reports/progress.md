# Version 1 Progress

This is an incomplete project, not a final performance report.

## Dataset

500 selected images from 25 videos; raw sources and curated images stay local.
The manifest, source hashes, annotations, and reports are tracked in Git.

| Split | Selected images | AI-reviewed images | Awaiting reviewed labels |
| --- | ---: | ---: | ---: |
| Training | 300 | 39 | 261 |
| Validation | 100 | 0 | 100 |
| Test | 100 | 0 | 100 |
| Total | 500 | 39 | 461 |

The 39 reviewed images contain 322 boxes and come from three training videos.
They are part of the 75-image training pilot, not an additional dataset. 36
pilot images and all 425 non-pilot images still require labeling and review. AI
self-review is not an independent annotation-quality audit.

## Implemented and Checked

- Raw-frame inventory and visual screening of 60 candidate sequences.
- Frozen source-prefix-isolated splits and 500 distinct sampled image hashes.
- Manifest-driven restoration without altering frozen metadata.
- Visual annotation guides, overlays, COCO export, and pending-label safeguards.
- Small custom backbone, anchor matching, box offsets, losses, mining, and NMS.
- CPU training, validation predictions, AP@0.5 metrics, freeze, and test commands.
- 21 focused tests for preparation, annotation gates, geometry, transforms,
  metrics, deterministic initialization, and controlled ablation settings.

## Training-Only Diagnostic

`pilot_overfit.json` records a 300-step CPU run on four reviewed training images.
Loss fell from 8.3101 to 0.0345 in about 21 seconds. The 0.5 operating threshold
produced 35 matches, two false positives, and no missed training reference boxes.

All four prediction overlays were visually inspected. Most reference boxes are
closely reproduced; image `01_01` contains an extra large prediction and a smaller
spurious prediction. Tiny targets remain difficult to inspect at model resolution.
This is memorization on four similar frames, not evidence of unseen-video quality.

## Remaining Before Version 1 Completion

1. Finish the 75-image pilot, review rules, and apply any frozen-rule corrections.
2. Label and review all remaining selected images; validate the completed dataset.
3. Run the full CPU baseline and controlled augmentation experiment.
4. Inspect validation predictions after each experiment and record error analysis.
5. Freeze the final validation-selected model and operating threshold.
6. Evaluate once on test data and write the final reproducible results report.

There are no validation metrics, final test metrics, or final model claims yet.