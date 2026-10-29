# Annotation Guidelines v0.1

Use these rules for the 75-image pilot only. After the pilot, revise this document once and freeze the resulting version before annotating the full dataset.

## Purpose

Create consistent, axis-aligned bounding-box labels for the vehicle-detection project.

## Label Set

Record the detailed class during annotation so the same dataset can support all project stages:

| Label | Include |
| --- | --- |
| `car` | Sedan, hatchback, SUV, minivan, van, pickup, taxi. |
| `truck` | Box truck, lorry, freight truck, semi-truck, dump truck. |
| `bus` | City bus, coach, school bus, articulated bus. |

For Stage 1, convert all three labels to `vehicle`. For Stage 2, convert `truck` and `bus` to `large_vehicle`.

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

If a vehicle cannot be reliably classified as `car`, `truck`, or `bus`, mark the image for review. Do not make a different decision for similar cases later.

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
