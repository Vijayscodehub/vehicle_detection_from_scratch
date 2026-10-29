# Annotation Guidelines v0.2 (Binary Pilot)

Use these rules for the 75-image pilot only. After the pilot, revise this document once and freeze the resulting version before annotating the full dataset.

## Purpose

Create consistent, axis-aligned bounding-box labels for the vehicle-detection project.

## Label Set

Use only `vehicle`: sedans, hatchbacks, SUVs, minivans, vans, pickups, taxis,
box trucks, lorries, freight trucks, semi-trucks, dump trucks, and buses.
Do not require a car/truck/bus subtype for the current binary detector.
Labels are AI-produced by visual inspection, not supplied BDD100K labels.

## Bounding-Box Rules

- Draw an axis-aligned box tightly around the visible pixels of the vehicle.
- Exclude shadows, reflections, and empty space.
- Label vehicles that are partly outside the image only when the class is clear.
- Label vehicles that are partly occluded only when the class is clear.
- Do not label vehicles smaller than 20 by 20 pixels in the original image.
- Label a truck with its attached trailer as one `truck` box when it is clearly one vehicle unit.
- Do not create a separate label for a trailer.
- Do not label bicycles, motorcycles, trains, emergency vehicles, or construction equipment in Version 1.

## Ambiguous Cases

If an object cannot reliably be identified as an in-scope vehicle, mark the image
for review. A clear vehicle whose subtype is uncertain can still be `vehicle`.

If the ambiguity would leave an in-scope vehicle unlabeled or make the image difficult to use, remove the image from the project dataset and record the reason in the manifest.

## Pilot Review Checklist

For every pilot image, check:

- Did I label every in-scope vehicle?
- Are boxes tight and consistent?
- Is every class decision supported by these rules?
- Is the image corrupted, too ambiguous, or unsuitable?
- Does this reveal a missing rule?

## Change Log

| Version | Date | Change | Reason |
| --- | --- | --- | --- |
| 0.1 | 2026-10-25 | Initial pilot rules | Test consistency before full annotation. |
| 0.2 | 2026-10-29 | Binary-only AI visual annotation | Authorized CPU-only Version 1 scope. |
