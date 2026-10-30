# Annotation Guidelines v0.3 (Frozen Binary Rules)

The 75-image training pilot has been reviewed. Version 0.3 incorporates the one
planned rule revision and is now frozen for the remaining dataset. If a serious
systematic defect is found later, stop labeling, document the change, and
re-review every affected annotation rather than silently changing the rules.

## Purpose

Create consistent, axis-aligned bounding-box labels for the vehicle-detection project.

## Label Set

Use only `vehicle`: sedans, hatchbacks, SUVs, minivans, vans, pickups, taxis,
box trucks, lorries, freight trucks, semi-trucks, dump trucks, and buses. Do not
require a car/truck/bus subtype for this binary detector. Do not label bicycles,
motorcycles, trains, marked emergency vehicles, construction equipment, or
pedestrians. Labels are AI-produced by visual inspection, not supplied BDD100K
labels.

## Bounding-Box Rules

- Draw one axis-aligned box per vehicle, tightly around its visible pixels.
- Do not merge adjacent or overlapping vehicles into one box.
- Exclude shadows, reflections, and empty space.
- Label a truncated or occluded vehicle only when the visible body is identifiable
	as an in-scope vehicle and the visible box is at least 20 by 20 pixels.
- Do not label vehicles smaller than 20 by 20 pixels in the original image.
- Label a truck with its attached trailer as one `vehicle` box when they are
	visually one vehicle unit.
- Do not create a separate label for a trailer.
- At dusk or night, label a vehicle only when its body outline is visually
	distinguishable. Headlights, tail lights, traffic signals, glare, and road
	reflections alone are not vehicles and must not receive boxes.

## Ambiguous Cases

If an object cannot reliably be identified as an in-scope vehicle, do not guess.
A clear vehicle whose subtype is uncertain can still be `vehicle`.

If an image's ambiguity or visibility makes it impossible to label every
in-scope vehicle consistently, exclude it and record the reason in the manifest.
Any replacement must come from the same split and retain the frozen video-level
assignment. Keep draft labels out of training until the whole image passes review.

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
| 0.3 | 2026-10-30 | Freeze binary rules after reviewing the 75-image pilot | Clarify single-object boxes, visible-size/low-light rules, exclusions, and handling of ambiguous images. |
